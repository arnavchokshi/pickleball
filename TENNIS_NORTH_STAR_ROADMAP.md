# TENNIS NORTH STAR — THE guiding document for extending the engine to tennis

**This is THE tennis guiding document.** It is to tennis what `NORTH_STAR_ROADMAP.md` is to pickleball:
the master TO-DO — what we build, in what order, and why — to take the single-camera
video→3D-world→AI-coaching engine we built for pickleball and open it up to tennis. It assumes the
pickleball North Star is **DONE** (the engine is mature, gate-passed, and shipping for pickleball);
everything below is the **delta** to make that same engine a first-class tennis product.

Owner-requested. Grounded in (a) a file-level audit of our own codebase's sport-genericity
(`runs/lanes/tennis_recon_20260707/` — what already generalizes vs. what is hardcoded pickleball) and
(b) a 26-dimension, ~60-agent adversarially-verified tennis SOTA research sweep
(`runs/research_tennis_20260707/` — ball, court, body, serve, racket, physics, coaching, product,
data, eval reports with sources).

**Status discipline (inherited verbatim from the pickleball roadmap — it is not negotiable):** this is
a PLANNING artifact. `CAPABILITIES.md` stays canonical on any truth conflict. **`TENNIS-VERIFIED=0`
today** — nothing below claims a passed tennis promotion gate. Checkboxes mean "work item exists",
never "capability verified". `[CORROBORATED]` marks a research claim that survived our adversarial
fact-check; it is NOT the reserved word VERIFIED (a passed PRODUCT gate on real tennis labels). All
standing rules in PART IV bind every task. Held-out/protected-surface eval labels are never touched
without a pre-registered ledger row — this is about truth, not law, and it carries over unchanged.

---

**The one-sentence thesis.** Tennis is *not* a green-field rebuild: the engine's hardest, most
expensive machinery — the 3D world (SAM-3D-Body meshes + smoothing + grounding + fusion), the ball
detector zoo, the calibration pipeline, the grounded-LLM coaching architecture, the iOS capture
stack, the data-engine + eval discipline — is either already sport-generic or transfers to tennis
*more cheaply than it did to pickleball*, because **tennis is the native training domain of the
tools we already use** (our ball anchor is literally a tennis checkpoint; the court detectors we
vendored were built on tennis). What tennis genuinely *adds* is a small set of new, high-value
pillars — serve biomechanics, spin, multi-surface physics, a broadcast-scale data engine — and a
much stronger competitive field. The work is real but it is **tuning + a few new pillars on a proven
chassis**, not a second car.

**Map of this document:** PART 0 owner-setup (the tennis ignition key) · PART I owner summary (I.0
what already generalizes vs. what is hardcoded · I.1 verdict · I.2 state-vs-bar · I.3 strategic calls
· I.4 phases · I.5 owner actions · I.6 first steps · I.7 definition-of-done / critical path / demo
milestones) · PART II tennis research verdicts (per pillar) + II-B delta matrix + II-C competitive/
market · PART III phase checklists (T0 sport-config foundation → T1 ball → T2 body → T3 racket → T4
court/net → **TS serve (NEW)** → **TF fusion** → TL live → T5 speed/QA → T6 coaching → T7 product) ·
PART IV standing rules · PART V evidence map · PART VI wave execution playbook.

**Naming convention:** tennis tasks use a `T`-prefix (`T0-x`…`T7-x`, `TS-x` serve, `TF-x` fusion,
`TL-x` live) so they never collide with the pickleball `P`-prefix in cross-references. Where a tennis
task is a near-clone of a pickleball task, it names the parent (e.g. "TS mirrors P6-3's reference-
range discipline").

# PART 0 — BEFORE THE TENNIS PROGRAM STARTS, OWNER MUST (read first; agents STOP here if any is blank)

A tennis run cannot begin until these are set. A manager agent that finds any missing does NOT guess
or work around it — it STOPS and surfaces the blocker (PART IV rule 9). Verify each against live
evidence before stopping; tick only WITH a dated evidence pointer.

- [x] **GPU access:** INHERITED from the pickleball program — same fleet mechanism (owner gcloud
  refresh token + SA impersonation; ≤$5/GPU/hr, max 4 GPUs, teardown-on-completion). Tennis adds no
  new GPU requirement; training corpora are larger (broadcast-scale) so budget a higher per-wave
  spend and state it in the boot prompt. No action unless auth has died (typed `needs-decision` STOP).
- [ ] **Sport-config decision (the tennis ignition key — BLOCKING before any tennis pipeline run):**
  the codebase already carries `Sport = Literal["pickleball","tennis"]` and a populated tennis court
  template, but the tennis path is unwired scaffolding (T0-1). This item is the owner's **go/no-go on
  the productization posture**: is tennis a (a) parallel product mode inside the existing app, or
  (b) a separate app/brand sharing the server core? This decides the iOS strategy (rename vs.
  parallel modules, T7-0) and the account/onboarding model. DEFAULT until owner rules: **(a) parallel
  sport mode on the shared server core**, iOS deferred behind server proof.
- [ ] **First tennis data — the owner capture ruling:** BROAD online-broadcast harvest is expected to
  be approved on the same basis as pickleball (private internal use, never redistributed; copyright
  waived as a concern) — but tennis broadcast has a real domain-gap caveat (broadcast camera ≠ our
  phone-on-court camera, the exact lesson that killed Roboflow pickleball transfer), so **owner
  in-domain captures remain the finisher**. Blank until the owner explicitly rules on (i) broadcast
  harvest OK, (ii) which surfaces the owner can capture on (hard/clay/grass — see I.5). No lane trains
  on tennis until this is ruled.
- [ ] **Multi-surface scope decision:** tennis is three sports physically (clay/grass/hard have
  different bounce, ball visibility, and line contrast). Owner ruling needed on **v1 surface scope** —
  DEFAULT: **hard-court first** (largest US/rec population, easiest visual), clay/grass as fast-
  follow. This gates the physics-constant and court-detector training scope (T1-4, T4-2).
- [ ] **Biometric consent decision** (unchanged from pickleball PART 0): gates PERSISTING any
  non-owner person's biometric profile (ReID gallery / shape betas). Session-only non-persistent
  tracking until answered. Tennis inherits this verbatim.
- [ ] **Any task-specific unblock** the first tennis wave needs (labeling, validation, a decision) —
  the manager lists these as typed STOPs at run start, per PART IV rule 9.

Everything below is the plan; this block is the ignition key. A blank field is a typed STOP, not a
proceed-anyway.

# PART I — OWNER SUMMARY (read this, skip the rest until you need it)

## I.0 What ALREADY GENERALIZES vs. what is HARDCODED PICKLEBALL (read this first)

The honest distinction this whole document rests on, from a file-level audit of the codebase
(`runs/lanes/tennis_recon_20260707/`): **the engine is a pickleball product with a genuine
dual-sport court-geometry library bolted underneath it.** The court-geometry layer is real two-sport
code; everything above it currently has no sport concept. Neither "pickleball with racketsport-
flavored names" nor "a finished multi-sport core" — it is a proven chassis with a tennis on-ramp
already half-built for pickleball's own reasons (multi-use-court line disambiguation + transfer-
learning off tennis SOTA).

### ✅ ALREADY SPORT-GENERIC (transfers to tennis with little or no code change)
- **`Sport = Literal["pickleball","tennis"]` threads through the pipeline.** `court_templates.py`
  already defines a **tennis court template** (78×36 ft, 36/42 in net, 21 ft service line) with real
  dimension math; `court_zones.py` has a tennis branch (service boxes + doubles alleys); `net_plane.py`
  is fully parameterized off `get_court_template(sport)`; `court_line_evidence.py` declares tennis
  line/net requirements; `orchestrator.py` threads `StageContext.sport` and hard-fails on
  calibration/context sport mismatch. A `--sport {pickleball,tennis}` CLI flag exists on
  `process_video.py`.
- **Player tracking + singles/doubles roles are already generic.** `doubles_id.py` and
  `player_global_association.py` assign sides/roles from track COUNT (`len==4` doubles, `len==2`
  singles) — tennis singles and doubles both fall out with **zero code change**.
- **The 3D-world spine is sport-agnostic.** SAM-3D-Body + MHR mesh, the person-masked camera-motion
  tracker, latent-space smoothing, foot-lock/grounding, and the Phase-F fusion optimizer are geometry
  + physics plumbing; sport-specificity flows in only through constants + the artifacts they consume.
- **The ball 2D detector is already a tennis model.** Our anchor checkpoint is
  `wasb_tennis_best.pth.tar` — a **tennis-trained WASB** reused zero-shot as the pickleball anchor.
  For tennis this pillar starts *closer to home than pickleball ever was*.
- **The data engine + eval discipline + grounded-LLM coaching architecture + speed/cost/QA infra**
  are all sport-agnostic frameworks; only their content (labels, reference ranges, shot vocabulary) is
  sport-specific.

### ⚠️ HALF-BUILT / UNWIRED (the tennis scaffolding exists but was never driven through a gate)
- `court_corner_review.py:164` — the tennis branch returns an **empty** `required_line_ids=()`,
  contradicting `court_line_evidence.py`'s tennis service-line requirement. Two modules disagree →
  the tennis path was never exercised end-to-end. (T0-1, T4-0)
- `ball_arc_solver.py:144` — `court_sport="pickleball"` is the default and is **not wired** from
  `orchestrator.StageContext.sport`. A tennis run silently solves arcs against the pickleball court.
  (T0-2)
- `schemas/__init__.py:30-45` — only `PICKLEBALL_COURT_KEYPOINT_NAMES` (15 points) exists; residual-
  length validation is hardcoded to 15 regardless of the `sport` field. There is no
  `TENNIS_COURT_KEYPOINT_NAMES`. The calibration gate only speaks pickleball. (T0-1, T4-1)

### ⬜ HARDCODED PICKLEBALL — must be replaced/added for tennis (this is the roadmap)
- **Ball physics** (`ball_arc_solver.py`, `flight_simulator.py`): mass 0.0255 kg, dia 74 mm, Cd
  0.33/0.45, Cl 0.195·S, restitution 0.58 / friction 0.16 — all pickleball-perforated-ball constants
  with **no sport dimension**. Tennis needs its own ball profile (~57 g, ~67 mm felt ball) and a real
  Magnus/spin model. **T1.**
- **Racket vs. paddle geometry** (`paddle_proxy.py`, `racket6dof.py`): the entire 6-DOF solver models
  a **flat rigid rectangle** (`X_paddle = W_hand ∘ G`) with per-paddle scanned corners. A tennis
  racket is an **oval strung frame on a long lever** — the PnP math is reusable but the rectangle
  silhouette + corner detection + grip priors all break. **T3.**
- **Shot taxonomy + rules + scoring** (`shot_taxonomy.py`, `shot_rules.py`, iOS rulebook copy): shot
  vocab (`dink`/`atp`/`erne`/`tweener`/third-shot), the kitchen/non-volley-zone, the two-bounce rule,
  USAP §7/§11 serve/NVZ faults, and USAP/DUPR 3.0–4.5 skill bands are pickleball-only and have **no
  `sport` parameter**. Tennis needs a wholesale substitution: groundstroke/volley/slice/overhead/drop/
  lob vocabulary, service-box legality, foot-fault, let, game/set/tiebreak scoring, NTRP/UTR bands.
  **T6 + a tennis rules module.**
- **Court keypoint detector head** (`court_keypoint_net.py`, `court_detector_v2_*`): built around the
  15 pickleball keypoints; needs a tennis output head + labels. **T4-2.**
- **iOS**: every target is named `Pickleball*` and there is **no `Sport` concept anywhere in Swift**;
  live guidance/court-dot-map is screen-space proxy only (nothing to port, nothing to reuse). **T7-0.**
- **A NEW pillar tennis demands that pickleball never needed: serve biomechanics (TS)** — the signature
  tennis coaching target (kinetic chain, toss, pronation, racket-head speed, foot fault, serve speed,
  injury markers). This has no pickleball analog and is a first-class new pillar. **TS.**

**One-line status:** the chassis is proven and the tennis on-ramp is half-paved. The work is (1) wire
the sport-config seam end-to-end, (2) swap the hardcoded pickleball constants for tennis, (3) add the
new tennis pillars (serve, spin, multi-surface, broadcast data engine), (4) re-earn VERIFIED on tennis
labels. `TENNIS-VERIFIED=0` until each does.

## I.1 The one-paragraph verdict

The engine already works and already speaks tennis at the layer that matters most. The court-geometry
core is genuinely dual-sport — a correct tennis court template (78×36 ft, 42″/36″ net, service boxes)
ships in `court_templates.py` today — player roles are format-agnostic (singles and doubles fall out
of track count with zero code change), and, the decisive inversion from pickleball, **the ball
detector is literally a tennis model**: TrackNet was invented on tennis, and our WASB anchor is a
tennis checkpoint. So the accuracy wall that defined pickleball — a holey-plastic ball no public
weight had ever seen, four measured zero-shot failures — largely evaporates: tennis 2D ball tracking
should reach bar with roughly an **order of magnitude less in-domain data** than pickleball demanded.
What tennis genuinely adds is (1) a strong, well-capitalized incumbent — **SwingVision**, single-iPhone,
on-device, ~$179.99/yr, a vendor-claimed ~500M-shot data moat, nothing like the weak pb.vision — and
(2) a handful of **net-new pillars pickleball never needed**: serve biomechanics (the sport's
signature coaching unit), spin estimation, multi-surface physics (clay/grass/hard, plus the clay
ball-mark as a free bounce label), a broadcast-scale data engine, and a real match-scoring FSM. The
competitive answer is *not* to fight on line-calling — Hawk-Eye/ELC owns officiating and SwingVision
owns cheap on-device calls, so we **cede both and stay explicitly advisory** — but to win on the one
thing no consumer rival ships: a **full 3D world (player mesh + 6-DOF racket + ball) from one
handheld/moving phone**, camera-motion tolerance that structurally unlocks the entire broadcast corpus
the fixed-camera field cannot ingest, multi-surface handling, and a zero-fabrication grounded-LLM
coach. Tennis also finally supplies ground truth pickleball never had — radar serve speed,
Hawk-Eye/clay-mark bounce, sensor-racket swing speed, marker/CalTennis pose — so several pillars can
earn their **first real VERIFIED gate**; but the richest GT is broadcast-domain and NonCommercial-
licensed, so the four-times-measured in-domain-data lesson still binds: broadcast *supplements*, owner
phone-capture *finishes*, and **TENNIS-VERIFIED stays 0** until a pre-registered gate passes on
our-camera labels. The plan: wire the sport seam end-to-end, swap the hardcoded pickleball constants
for tennis, stand up the new pillars, and re-earn VERIFIED — tuning + a few new pillars on a proven
chassis — with one hard gate before any paid launch: a freedom-to-operate opinion on SwingVision's
granted US 11,893,808 B2 (monocular NN 3D ball extraction on a mobile device).

## I.2 Where tennis stands vs. the bar (all numbers sourced)

The table below is the tennis analog of the pickleball state-vs-bar matrix. "Our platform on tennis
day-1" is what the *existing engine* produces the moment the sport seam is wired (PART III T0), before
any tennis-specific accuracy work. Competitor figures are cited inline with their verification status
(vendor-claim vs. independently measured); the adversarial sweep's corrections are already applied
(e.g. Hawk-Eye's documented error is the **3.6 mm advertised average** — the oft-cited "2.2 mm" has no
primary source; camera count is ~6–12, up to ~18, not "10–18"; SwingVision's "~97%/10 cm" and
"500M shots" are vendor figures, not audited). Full per-pillar reasoning is PART II; the full delta
matrix is PART II-B.

*Competitor numbers cited from corpus. Vendor/marketing claims labeled as such; independently-measured figures noted where the corpus verified them.*

| Capability | Our platform on tennis day-1 | Competitor bar (with numbers) | Tennis target gate |
|---|---|---|---|
| **Court keypoint / homography** | Reimplement yastrebksv arch (TrackNet-like, 15 ch) on our license; pre-train on 8,841 broadcast imgs → fine-tune our-camera. Metric-15pt PnP already at 8 px/15 px on pickleball | **yastrebksv TennisCourtDetector**: full postproc 96.3% prec / 96.1% acc / **1.83 px median** @7 px; base model 93.6/93.3/2.83 px. All on *broadcast* val. No license file (all-rights-reserved) | PCK@5px ≥ 0.95 per viewpoint; surveyed court-corner reprojection ≤ 5 cm; report per-surface (hard/clay/grass) on our-camera held-out |
| **Ball 2D tracking** | WASB (MIT) tennis checkpoint drop-in; ~0.90+ F1 expected on broadcast-like views | Native TrackNet tennis test: F1 **0.90–0.97** (V2 0.9037–0.9677); **RacketVision MS-TrackNetV3** tennis MDE **1.96 px**, mAP 81.9, P 0.945 / R 0.880; TrackNetV5 0.9859 (no code/weights) | Broadcast-view F1 ≥ 0.92 (near-free); **our-camera phone F1 ≥ 0.85** after modest fine-tune (vs. pickleball's 0.70 zero-shot wall) |
| **Ball 3D flight / serve speed** | Retune ODE for felt-ball Cd/Magnus; serve-speed via homography + drag; **currently no spin, 35 m/s ceiling** | **SwingVision**: shot speed within **±10%** at 60 fps (vendor); **Hawk-Eye/ELC**: ball position ~**3.6 mm** advertised avg, 6–12 cams (up to ~18) @340 fps (unreachable single-cam); broadcast **radar ±1 km/h** | Serve-speed error ≤ 5% / ±3 mph below 100 mph, degraded band above; 3D-arc reprojection ≤ detector noise; **advisory only** |
| **Spin (topspin/slice/kick)** | Class-first via BST/uplifting-net over 2D track+bounce; coarse 3-band RPM; direct RPM only serves @240 fps | **SwingVision** spin recognition **ICC=0.76** (n=5 elite juniors — small sample); **UpliftingTT** binary topspin/backspin 97.1% synthetic / **89.5% real** (table-tennis only); event-camera axis err **33°** | Spin *class* macro-F1 ≥ 0.85; coarse RPM band ≥ 80% vs. radar GT; serve RPM ≤ 8% MAE @240 fps; 3D axis NOT gated (advisory) |
| **In/out line call** | σ_bounce advisory + `too_close_to_call`; needs ≥120–240 fps | **SwingVision**: **~97% within 10 cm** single-cam @60 fps, **>99% overall**, 99% with 2nd cam (vendor); 500+ USTA matches officiated 2024. **Hawk-Eye ELC**: ITF-certified ~3.6 mm, ATP-tour standard 2025, Wimbledon 2025 | Clear-call agreement ≥ 0.95, **zero confident-wrong**; every call carries σ band; **ADVISORY, never officiating** (we cede ELC) |
| **Player tracking / ReID** | YOLO26m + BoT-SORT + OSNet reuse; singles = easy 2-gallery; tiled crop for far player | **Deep-EIoU** SOTA on SportsMOT; **GTA** HOTA **81.04** (SportsMOT), +10.24 HOTA over SORT; SoccerNet-GSR winner GS-HOTA 63.81 | HOTA ≥ 75 online / ≥ 80 after GTA stitch; doubles ≤ 1 IDSW/player-min; far-player recall floor ≥ 90% @≥24 px box |
| **3D body / pose** | Frozen SAM-3D-Body + MHR + classical grounding; **first-ever GT validation** via CalTennis | **CalTennis** (11M frames, CC-BY-**NC**): SOTA PA-MPJPE **~84 mm** (PromptHMR) — local pose largely solved; **world translation 0.9–3.6 m** — unsolved monocular; **AthletePose3D** fine-tune cuts athletic MPJPE **214→65 mm** | PA-MPJPE ≤ 70–85 mm; world foot-position ≤ 0.30 m (court-grounded, beat PromptHMR 0.94 m); airborne-foot: 0 phantom pins |
| **Racket 6-DOF** | Full fused estimator reused; RacketVision 5-kp seed + tennis geometry | **RacketVision** (MIT): racket PCK@0.2 **89.6%**, MPJPE **5.34 px** — but face-width (left/right) keypoints only **~80%** vs. >92% structural; **TT4D** physics-inversion face angle **26.4±4.4°** | Racket PCK@0.2 ≥ 0.85 overall **and ≥ 0.80 on face-width**; face-normal ≤ 15° (beat TT4D 26° inversion ceiling); rotation jitter ≤ 5°/frame |
| **Stroke recognition** | New pose+trajectory pillar (BST port) + PoseC3D backbone | **BST** (ShuttleSet, badminton): 83.22% top-1 / 0.8097 macro-F1 (25-cls); **THETIS** tennis video-only CNN-LSTM **79.17%** (12-cls); **SwingVision** stroke detection **ICC=0.97** (still swaps volleys↔flat FH per user reports) | Coarse 4-class (serve/FH/BH/volley) macro-F1 ≥ 0.85; 1H-vs-2H BH ≥ 0.80; topspin/slice ≥ 0.70; beat THETIS 79.17% baseline with 3D features |
| **Serve biomechanics** | 8-stage segmenter + proximal metrics (knee/hip/trunk, contact height) honest; distal advisory | No consumer competitor ships kinematic-chain coaching; **Kovacs-Ellenbecker** literature norms (max ER 172±12°, GRF 1.68–2.12 BW); **OpenCap** validated 2.0–10.2° (proximal only) | Contact-frame ≤ 1 frame @240 fps; kinematic-sequence order ≥ 90%; proximal angular-vel within ~10–15%; distal/racket-head **NOT gated** until racket-fusion |
| **Multi-surface** | Surface classifier + CPR-indexed bounce presets; clay ball-mark spike | **Foxtenn** (ITF Gold) real-image clay bounce (40-camera rig); ATP uses Hawk-Eye ELC on clay 2025 (Madrid/Rome) | Surface classify ≥ 97%; per-surface bounce-height ≤ 10%; clay ball-mark advisory ≥ 85% agreement (explicitly NOT vs. Hawk-Eye) |
| **Coaching / LLM** | Full grounded 3-stage engine; tennis reference library from MCP/ITF; the layer **no competitor ships** | **SwingVision/Hawk-Eye stop at stats/charts** — none give grounded NL coaching; *Talking Tennis* validates pattern (3 coaches rated feedback 7.3–9.3/10; 100% no-fabrication over 317 *isolated-stroke* outputs) | 0/300 fabrication audit; coach rubric ≥ 8/10; 100% reference-range provenance; every stat trust-banded |
| **Match scoring** | Greenfield config-driven FSM (deterministic) | No CV competitor exposes a full ITF-variant scoring engine as product | 100% exact on scripted point suites (deuce/No-Ad/tiebreak/Coman/best-of-3/5); validate vs. MCP scorelines |
| **On-device live** | L0/L1 advisory parity target; cloud L3 as premium hook | **SwingVision**: real-time **on-device ANE**, locked 1080p/60 fps, ~zero marginal cost, "won't see the bounce below 60 fps"; iPhone 11+ sustained (thermal unverified in corpus) | Ball every-frame @60 fps (0 skipped); live loop p90 < 16.6 ms/frame; ≥90-min outdoor thermal soak; 240 fps capture-only |
| **Business / GTM** | Larger TAM + institutional B2B + broadcast pretraining corpus | **SwingVision**: **$179.99/yr**, **20,000+ subs**, **~$2.75M 2024 rev**, $6M Series A (Oct-2023, Sony-backed), 100 college teams; **SportAI** $3M round + MATCHi 17k courts; tennis academy economy ~$8.4B | Price $149–179/yr; free→paid ≥ 3–5%; LTV:CAC ≥ 3:1; per-session cloud L3 COGS < 15% ARPU; ≥10 academy design partners for in-domain data |

**Notes on Table B.**
- **Where we do NOT compete:** line-calling accuracy/latency. SwingVision owns cheap on-device advisory calls (~97%@10 cm vendor) and Hawk-Eye owns ITF-certified officiating (~3.6 mm, $40–100K/court). Our wedge is **single-phone full-3D-world + grounded coaching + camera-motion tolerance + multi-surface**, positioned honestly advisory.
- **Verification caveats carried from the corpus:** SwingVision's "~97%/10 cm", ">99%", and "500M shots / 200k players" are **vendor/marketing** figures, not independently audited. Hawk-Eye's oft-cited "2.2 mm" is unverified — only the **3.6 mm** advertised average is documented; camera count is **~6–12 (up to ~18)**, not 10–18. AthletePose3D's 214→65 mm is on **mixed sports** (12 motions), not tennis — CalTennis is the tennis validator. CalTennis is **CC-BY-NC** (NonCommercial) — not shippable as-is.
- **The recurring binding constraint** (measured 4× on pickleball) survives tennis's data abundance: broadcast/Hawk-Eye/sensor GT is **REFERENCE-VERIFIED** and transfers through 3D geometry, but only **owner phone-capture (OUR-CAMERA-VERIFIED)** data can back a consumer-tier claim. Broadcast abundance must *supplement*, never *replace*, in-domain capture.
- **The two structurally hardest, most likely-to-stay-UNVERIFIED targets:** distal serve kinematics / racket-head speed (shoulder IR ~2400°/s — worst monocular signal) and precise per-shot spin RPM (sub-frame at 30–60 fps; needs owner radar GT). Both ship advisory-banded until a real-label gate passes.

## I.3 The strategic calls for tennis (from research + our code audit)

These are the load-bearing decisions the 26-dimension tennis research sweep + the codebase audit
produced. Each is grounded in PART II's per-pillar evidence; where a "known fact" died in adversarial
verification it is corrected here (e.g. Roland Garros did **not** adopt ELC in 2025; the 2021-era
"Vid2Player" is 2D-only — only Vid2Player3D does 3D, and its models are withheld). The eight calls,
then the enumerated new pillars tennis adds beyond the pickleball P0–P7 set.

**1. Ball tracking transfers cheaply because tennis is the ball tracker's native domain — but "cheap" is not "free," and the transfer is broadcast-domain.** TrackNet (Huang et al., 2019) was *invented on tennis broadcast video*, and every strong successor ships a tennis checkpoint: WASB-SBDT (MIT, nttcom, BMVC 2023) and TrackNetV3 (github.com/qaz812345) both carry tennis weights, and WASB is already the standing best ball stage on our pickleball stack. This inverts the pickleball wall (best held-out ball F1 ~0.6969 vs a 0.7248 bar, ledger rows 22–23, driven by a holey-plastic ball no public weight had ever seen): swapping to upstream tennis WASB/TrackNetV3 checkpoints should clear ~0.90+ F1 on broadcast-like views near-free. Two honesty guards apply. First, keep the eval regimes distinct — the RacketVision reproductions (P0.94/R0.80–0.88, implied F1 ~0.86–0.88) are a *different* test set from the native TrackNetV2 tennis set (F1 reported 0.9037 in the WASB paper up to 0.9677 in the TrackNetV5 reproduction, split- and tolerance-dependent), so pin the split and 4px tolerance in our own gate. Second, every one of these numbers is on high broadcast angles; our courtside iPhone view remains out-of-distribution, so a modest fine-tune (hundreds-to-low-thousands of frames) is still needed — but that is roughly an order of magnitude less than pickleball demanded. Do **not** plan around TrackNetV5 (arXiv 2512.02789, headline F1 0.9859): its code ships under an explicit **proprietary all-rights-reserved SDK license (Shanghai Code Zero)** with weights and datasets withheld, so it is unreproducible/watch-only. Retire TrackNetV4 — its motion-fusion checkpoint is undeserializable upstream and it even scores *below* V2 in the V5 reproduction.

**2. The broadcast/Vid2Player corpus relocates the data bottleneck rather than dissolving it — the two-flywheel doctrine.** Tennis has a genuinely huge, court-anchored corpus (TennisVL/TennisExpert: 202 broadcast matches, 471.9 h, 40,523 rally clips, arXiv 2603.13397), and broadcast's fixed elevated known-court geometry gives near-free homography that pickleball never had — real 3D ball arcs, free serve-speed OCR, pro reference distributions. But the central adversarial finding holds: broadcast is a **different appearance domain** from a fence-mounted low iPhone (tiny motion-blurred ball, different player/racket appearance), which is exactly the Roboflow-transfer death measured 4× on pickleball. The competitive proof is decisive — SwingVision's edge reportedly comes from ~500M shots captured by ~200k users' *phones*, an in-domain flywheel, not broadcast. And Vid2Player is widely mis-cited: the 2021 TOG paper is **2D controllable video sprites with no 3D reconstruction at all**; only Vid2Player3D (SIGGRAPH 2023, NVIDIA) does 3D, and it ships under a non-commercial **NVIDIA License** with trained models *withheld for broadcast-footage copyright*, so it is a method blueprint, not a reusable asset — and its physics-avatar output (unlimited joint torques + residual root force) is generation, not measurement-grade kinematics. The strategy is therefore two flywheels: **broadcast for physics/geometry/motion/reference priors that transfer through 3D court geometry; owner/coach phone-capture SST for the detector/appearance domain that does not.** Fund the in-domain capture from day one; do not let broadcast abundance de-fund it.

**3. SERVE is a new flagship pillar — and it is precisely where the single-camera stack is strongest and weakest at once.** The serve is tennis's signature coaching target, well-quantified by the Kovacs–Ellenbecker 8-stage model (max external rotation 172±12° reached ~0.09 s pre-contact; contact racket velocity 38–47 m/s; peak vertical GRF 1.68–2.12 BW; legs+trunk 51–55% of energy) and Elliott's ITF corpus. The stack already has the scaffolding (serve is a `SHOT_TYPE`; contact fused from wrist-velocity + ball-inflection + audio). But the distal velocities that generate racket-head speed — peak shoulder internal rotation ~42 rad/s (~2400°/s) and wrist flexion ~34 rad/s (~1950°/s, per Sakurai 2013) — are the joints monocular HMR captures *worst*, and the acceleration phase (max-ER→contact) unfolds in **<0.01 s**, unsampleable at 30–60 fps. Two rulings follow: serve analysis **must** run on the 240 fps sidecar mode, and **distal angular-velocity / racket-head-speed claims stay trust-banded or out-of-scope until the racket-6-DOF retool + a validated gate exist** — never presented as measured. What *is* honestly deliverable now is proximal (leg drive, knee flexion, hip/trunk rotation, trophy-pose statics, contact height, kinematic-sequence *ordering*), where markerless validates well (OpenCap RMSE 2.0–10.2°; an HRNet markerless study matched marker-based external work to within 7.3/9.3 J). Injury-risk output must remain non-clinical advisory — video cannot measure shoulder distraction (~0.5–0.75 BW) or ~300 N·m deceleration torque, and stating them as load is a medical-liability trap.

**4. SwingVision is a far stronger incumbent than pb.vision — cede line-calling, win on the 3D world + grounded coaching.** SwingVision is single-iPhone, real-time on the Neural Engine, ~$179.99/yr (~$2.75M 2024 revenue, 20,000+ subs, ~$10M raised incl. a $6M Series A Oct-2023, Apple Editors' Choice, 500+ USTA matches in 2024), with vendor-claimed ~97% on close calls within 10 cm — and it already added pickleball, so it can attack our niche downmarket. Its on-device inference is ~zero marginal cost; our only gate-passing depth tier (SAM-3D-Body) is cloud GPU. Competing on line-calling latency/accuracy is a guaranteed loss. Our genuine wedge is the combination no consumer rival ships: a **full 3D world (player mesh + racket 6-DOF + ball) from one handheld/moving phone**, **camera-motion tolerance** (ARKit pose) that structurally unlocks the broadcast corpus the entire fixed-camera field (Hawk-Eye, PlaySight, Wingfield, Zenniz, Baseline) cannot ingest, **multi-surface** handling, and the **3-stage zero-fabrication grounded-LLM coach** that all competitors stop short of. Positioning rulings: brand every line/serve/foot-fault call **advisory, never officiating**; make a *mesh-only* coaching insight (kinetic-chain sequencing, contact-point-in-3D, racket-face angle at impact) that a 2D overlay provably cannot produce, so "3D" is a felt benefit not a demo word; and anchor pricing at ~$150–180 with a B2B academy/college/coach-seat engine ($8.4B academy economy) that SportAI is already racing to lock via MATCHi's 17k courts.

**5. Multi-surface is a genuinely new axis — physics, vision, and a clay-only free label.** Pickleball is effectively single-surface; tennis is clay/grass/hard, and the counterintuitive core fact must be respected: clay is "slow" from high friction (μ≈0.8) that scrubs pace yet bounces *highest* because its vertical restitution is largest (ITF averages ≈0.83 clay / ≈0.80 hard / ≈0.73 grass); grass is "fast" from low friction (μ≈0.6) that skids the ball low. Adopt the ITF **Court Pace Rating** (CPR = 100(1−μ) + 150(0.81−e_T), 5-category slow→fast) as the physics knob for both the surface-conditioned bounce residual and coaching ranges — and keep CPR (lab pendulum) distinct from Hawk-Eye's in-match CPI. Three new builds: a cheap surface classifier (red-orange clay / blue-green-gray hard / green grass) selecting physics + ranges; surface-robust ball detection (optic-yellow contrast collapses on grass — the historical reason balls went yellow in 1972 — and felt stains reddish on clay); and the flagship differentiator, **clay ball-mark detection**, a physical skid mark giving a near-exact 2D bounce point unavailable in pickleball or any other surface. Frame ball-marks as advisory "see your bounce," never officiating — a single non-ground camera cannot match Foxtenn's ~40-camera clay rig.

**6. Hawk-Eye/ELC is the officiating gold standard and the pro norm — stay explicitly advisory, and get the milestones right.** ELC is officiating infrastructure: ~6–12 (up to ~18) tracking cameras per court at 340 fps, ~3.6 mm advertised mean error (the oft-cited ~2.2 mm figure has no primary source and should be dropped), ITF-certified since 2006. The realistic single-camera ceiling is SwingVision's vendor-claimed 97%-within-10 cm — the aspirational advisory bar, not a gate. Two factual guards the corpus flagged: **Roland Garros did *not* adopt ELC in 2025** — it remained the lone Slam using human line judges (through 2026); the correct 2025 milestone is the **ATP mandating ELC at all events including clay (Madrid, Rome)**, which produced public ball-mark-vs-digital disputes. And Foxtenn is the pioneering clay mark-verification system, **not the sole tour-approved clay ELC**. Our honesty discipline (`too_close_to_call`, σ_bounce band per call, advisory-not-officiating) is now table stakes, not a differentiator — and the officiating-claim boundary is also a legal one (see call 8).

**7. Racket 6-DOF geometry changes shape, but tennis finally supplies real racket ground truth — with weak spots to respect.** The pipeline transfers at the architecture level: the IPPE planar-PnP core is *valid* because a strung face is a coplanar ellipse with a well-defined normal, and our 5-keypoint schema ≈ RacketVision's top/bottom/handle/left/right (RTMPose-M, **MIT**, AAAI 2026 oral). What breaks is geometry and priors: swap the flat 16×8″ rigid rectangle for a **~27″ elliptical strung face on a long lever** (string-bed give ~10–30 mm, so the rigid 37 mm impact cap is wrong), add **one/two-handed backhand dual-wrist grip** anchors, replace the single continental grip prior with a tennis grip taxonomy, and rebuild swing-speed as **v_head = v_wrist + ω×r_lever** (head 30–45 m/s at serve, far above wrist speed). The GT news is real but bounded: RacketVision's face-**width** keypoints (left/right) are the weak axis at ~80% PCK vs >92% structural (occlusion+blur), so gate on them explicitly; ball-impact inversion alone is capped at ~26° face-angle error (TT4D mocap validation, 140+ h, Kienzle et al.), so it stays advisory; and sensor rackets give strong **swing-speed** GT (Zepp racket-speed ICC 0.983 vs 500 Hz VICON) but **weak impact-location/spin** GT (Zepp κ_w=0.217) — and the hardware (Babolat Play, Sony STS, Zepp) is **discontinued 2020–2021**, so it is a legacy calibration rig, not a supply.

**8. Tennis's ground-truth abundance de-risks VERIFIED — most where pickleball is weakest — but the richest GT is broadcast-domain and NonCommercial-licensed.** Pickleball sits at VERIFIED=0 with zero real 3D labels for ball flight, body pose, or racket; tennis has physically-calibrated GT for nearly every pillar and, for the first time, enables gates pickleball could never define: **serve-speed error vs broadcast radar** (radar itself ±1 km/h), **3D bounce-location error vs Hawk-Eye / clay ball-mark**, **court-corner reprojection vs ITF-surveyed geometry**, and the **first real BODY gate** — world-MPJPE vs marker-based serve mocap and against **CalTennis** (Caltech; SOTA local pose ~84 mm PA-MPJPE but 0.9–3.6 m world-translation error, foot-contact unreliable). The binding caveats: CalTennis is licensed **CC-BY-NC-4.0 (NonCommercial)** — usable for internal eval, not as commercial training data; **AthletePose3D** is a general 12-sport, non-commercial benchmark (its 214→65 mm fine-tune gain is *not* tennis validation); the Match Charting Project (17,808 matches / ~10.5M shots) and tennis_atp/wta are **CC BY-NC-SA** (derive aggregate ranges, never redistribute rows); the canonical **yastrebksv/TennisCourtDetector ships no license file** (default all-rights-reserved) and the Gholamreza HF "MIT" mirror is a re-upload of the *same* all-rights-reserved broadcast frames, so an MIT tag confers no valid commercial rights; and Hawk-Eye/SkeleTRACK 3D data is partnership-gated, not a public download. Two correction guards for the write-up: yastrebksv's headline 96.3%/1.83 px is the *full* refine+homography config (raw base is 93.6%/2.83 px), and arXiv 2511.04126 is a **custom ResNet50 at ~3.8 px that does not reuse yastrebksv** — its "1.83 px identical" claim was false and must not be repeated. Operationally, tag every metric earned on broadcast/Hawk-Eye/sensor GT **REFERENCE-VERIFIED**, distinct from **OUR-CAMERA-VERIFIED**; only the latter backs a consumer claim. And IP is now live-fire: SwingVision holds **US 11,893,808 B2** (monocular NN 3D ball extraction on a mobile device, priority 2020) and was sued over Infinity Cube's **US 10,467,478** (S.D. Cal. 3:22-cv-00547) — a written FTO opinion on both is a launch-blocking gate, and NonCommercial training data must be quarantined from any shipped model.

---

### New pillars tennis adds (beyond the pickleball P0–P7 set)

Each is genuinely net-new — either no pickleball code analog exists, or the pickleball assumption is structurally invalid.

1. **Serve-biomechanics pillar.** An 8-stage temporal segmenter (start/release/loading/cocking-trophy/acceleration/contact/deceleration/finish) anchored on the fused contact instant, gating foot-lock, adaptive smoothing, and kinetic-chain feature windows; hard 240 fps entry condition; a signed, coach-reviewable serve reference-range library (Kovacs 8-stage + Elliott ITF). Highest-value coachable unit in the sport; no pickleball analog. *Ships proximal metrics first; distal velocity + racket-head speed trust-banded/out-of-scope until the racket retool.*

2. **Multi-surface adaptation pillar.** Surface classifier (clay/grass/hard) → surface-indexed bounce model {vertical COR e_v, tangential friction μ, spin-coupling / grip-slide regime per Cross} seeded from ITF CPR presets, plus surface-conditioned ball detection and coaching ranges. Pickleball is single-surface; this touches Ball, Fusion, Court, and Coaching.

3. **Clay ball-mark review (sub-pillar of #2, but architecturally distinct).** Segment the physical skid mark, fuse with the trajectory-derived bounce point for a mark-anchored advisory in/out + skid-length (spin/angle proxy). A physical ground-truth mechanism with **zero analog in pickleball or on any other surface** — the sport's own review culture, digitized.

4. **Spin-estimation pillar.** Net-new capability — the ball-3D schema's `spin_rpm` is currently hardcoded `None`. Two-stage trajectory-inverse (2D track + bounce → transformer uplift trained on tennis MuJoCo synthetics, retargeting UpliftingTableTennis by **Kienzle, Ludwig, Lorenz, Satoh & Lienhart**, WACV 2026, GPL-3.0) for spin *class* (topspin/backspin/slice/flat) + coarse RPM band; a high-fps periodic-signal path for precise serve RPM. Class is gate-able on human labels; any RPM number needs owner radar GT.

5. **Broadcast data-engine pillar.** A distinct ingestion subsystem (no ARKit sidecar, moving PTZ camera): YouTube ingest → rally/shot segmentation → auto-homography court fit → WASB/TrackNet ball → trajectory-kink+audio bounce → scoreboard/serve-speed OCR, producing physics/geometry/reference priors and SST pretraining targets. Footage never redistributed; models kept off any NonCommercial/broadcast-copyright path.

6. **Singles/doubles rules & match-scoring engine.** A greenfield deterministic FSM (points→game→set→match, deuce/advantage, tie-break serve rotation, config-driven Appendix VI variants) — the repo has **no** rally/score/deuce state today — plus score-driven diagonal service-box validity, singles-vs-doubles sideline/alley selection, service-fault/let/double-fault sub-FSM, and a foot-fault advisory. Deterministic and 100%-testable; the one tennis subsystem that can be VERIFIED to exactness with zero ML.

7. **Serve/returner temporal-role & stroke-recognition pillar.** A per-point server/returner role bound to the serve event + deuce/ad court, and a **pose+ball-trajectory** stroke recognizer (BST-style cross-attention over MHR skeletons + arc trajectory) that replaces the pickleball ball-geometry-only classifier — because FH vs BH, one- vs two-handed backhand, and topspin vs slice are **pose/swing phenomena, not ball-flight buckets**.

8. **Match-scale orchestration (speed/cost).** A match-level "rally splitter" in front of the per-clip pipeline (a best-of-5 match is up to ~1.08M frames but only ~10–25% ball-in-play), client-side rally-gated upload (raw 4K60 match = 14–55 GB), per-rally QA isolation and checkpoint/resume, and mandatory batched warm-pool BODY inference — a prerequisite, not a stretch, given SwingVision's ~zero on-device marginal cost.

*Racket 6-DOF, Body/mesh, Court calibration, Ball flight, Fusion, and Coaching are **not** new pillars — they carry over structurally and are re-parameterized (felt-ball aerodynamics with contested Cd — Goodwill/Cross 0.507 vs Mehta 0.6–0.7, banded; strung elliptical racket; tennis court template; surface-aware bounce; tennis reference ranges), not rebuilt.*

## I.4 Phases at a glance

Tennis reuses the pickleball phase spine and adds one new pillar phase (**TS — Serve**). `T`-prefix
throughout. Multi-surface, spin, the broadcast data engine, and the scoring FSM are folded into the
phases they most touch (T4, T1, T0, T6) rather than given their own top-level phase, and are called
out as new-pillar work in PART III.

| Phase | Outcome | Gate that ends it | Runs in parallel with |
|---|---|---|---|
| **T0 Sport-config foundation** | `--sport tennis` computes tennis geometry end-to-end; first honest tennis 3D world; tennis data ingest + surface tagging + held-out ledger; broadcast-harvest spike | Seam wired (keypoint schema, `court_sport` threaded, tennis physics stub) + fresh tennis E2E bundle on ≥1 clip, all bands honest, **pickleball suite green** | — (everything depends on it) |
| **T1 Ball (2D transfer → 3D flight → spin)** | Tennis ball to bar cheaply (transfer bet); serve-speed regime (fps/teleport/blur); true felt-ball 3D flight; spin *class* + coarse RPM | Our-camera F1 ≥ 0.85; serve-speed ≤ 5% vs radar; spin-class macro-F1 ≥ 0.85; 3D-arc reproj ≤ detector noise | T2, T3, T4, TS |
| **T2 Body (tennis-dynamic hardening + first real BODY gate)** | Serve/open-stance/lunge robustness; airborne-foot fix; racket-occlusion; **first-ever BODY GT validation** (CalTennis/mocap) | PA-MPJPE ≤ 70–85 mm; world foot-pos ≤ 0.30 m; 0 phantom foot-pins on jump serves | T1, T3, T4 |
| **T3 Racket 6-DOF (paddle→racket)** | Strung-elliptical-face model, long-lever swing speed, grip taxonomy, sensor-racket GT | Racket PCK@0.2 ≥ 0.85 incl. ≥ 0.80 face-width; face-normal ≤ 15°; swing-speed vs sensor GT | T1, T2 |
| **T4 Court/net + multi-surface** | Tennis keypoint detector, singles/doubles classifier, piecewise net, surface classifier + surface-indexed bounce + clay ball-mark | PCK@5px ≥ 0.95 per surface; surveyed-corner reproj ≤ 5 cm; surface classify ≥ 97%; per-surface bounce ≤ 10% | T1, T2, T3 |
| **TS Serve (NEW PILLAR)** | 8-stage segmenter, proximal kinematics + serve speed + foot-fault; distal velocity/racket-head-speed advisory-only | Contact-frame ≤ 1 frame @240 fps; kinematic-sequence order ≥ 90%; proximal angular-vel within ~10–15% | after T1/T2/T3 partials |
| **TF Global fusion (tennis speed)** | One mutually-consistent metric tennis world — ball meets strung face, feet planted (except jump serve), surface-coupled bounce | Impact-gap + floor-penetration decrease vs standalone; cross-system consistency asserted; ≥120–240 fps precondition met | after T1–T4 partials |
| **TL Live tier (advisory)** | On-device tennis: court lock, serve/stroke overlays, advisory in/out + serve speed; ball decoupled from cadence scheduler | Live-loop p90 < 16.6 ms/frame; 0 skipped ball @60 fps; ≥90-min outdoor thermal soak | continuous parallel stream |
| **T5 Speed/cost/scale** | Match-scale orchestration (rally splitter, score OCR, checkpoint/resume); per-match COGS metered | Timed rally-gated match E2E; deep-session L3 COGS < 15% ARPU (~$1–3/session) | continuous |
| **T6 Coaching (tennis)** | Tennis shot taxonomy (pose+trajectory), stat layer (serve %, W:UE, BP), UTR/NTRP reference library, grounded coach, serve breakdown | 0/300 fabrication audit; coach rubric ≥ 8/10; 100% reference-range provenance; self-relative framing | after T1/T2 partials |
| **T7 Product (tennis mode)** | iOS sport strategy (per PART 0 ruling), onboarding, pricing vs SwingVision, **FTO + no-officiating-claims** | Real-device tennis E2E + FTO opinion cleared + NonCommercial-data quarantine audited | last |

## I.5 What only YOU can do (highest-leverage owner actions for tennis, easiest first)

Tennis reuses the pickleball engine, so the owner-only work is smaller than pickleball's was — but it
is different in three ways: **surfaces, the serve, and ground-truth abundance you can tap.**

1. **Record tennis on your home court(s).** Tripod (≥7 ft — a bigger court needs more height to see
   both baselines + all lines), whole court + every line + both baselines in frame, landscape,
   **1080p60 minimum; use 120/240 fps for serve/contact clips** (serves are 2–4× faster than a
   pickleball ball — 60 fps blurs the contact). Audio ON. **Singles AND doubles** (both fall out of the
   generic role code, but we need eval clips for each). A handful of deliberately-handheld clips =
   the motion-tolerance test set. Unblocks T1, T3, T4, TS.
2. **Multi-surface captures (the tennis-only axis).** If you can reach them, one session each on
   **hard / clay / grass** — even a single short clip per surface. Your PART 0 surface-scope ruling
   rides on what you can capture. **Clay is especially valuable**: the physical ball-mark it leaves is
   a unique, shippable in/out-review feature (T4 multi-surface).
3. **Serve/spin ground-truth session (a tennis superpower pickleball lacked).** Wear a **sensor racket**
   (Babolat Play / Zepp / Sony Smart Tennis Sensor) for a serving + groundstroke session → swing-speed,
   spin (RPM), and impact-location GT for T3/TS/spin. If you can film serves next to a **radar reading**
   (a serve-speed display, a cheap radar gun, or a court that shows it) → ball-speed GT. These retire
   entire pillars' worth of "we can't validate it" risk cheaply.
4. **Racket pose GT.** 4 edge/corner markers (tape) on your racket + one slow-mo orbit video of it
   (doubles as the reference scan) → the only path to RKT tennis-VERIFIED (T3-7).
5. **The PART 0 rulings** (each unblocks a chunk of the roadmap): sport-config posture (parallel mode
   vs. separate app), v1 surface scope, broadcast-harvest go/no-go.
6. **One-time profile captures per court** (~an afternoon): ChArUco lens sweep per zoom preset;
   empty-court clip per court+surface; **tape-measured net heights (posts AND center — tennis is
   42 in / 36 in)**; your NTRP/UTR level (the skill-band anchor for coaching reference ranges).
7. **(Optional, high value) "called-outcome" clips.** Film a set of serves/points where you call the
   outcome out loud (ace/fault, winner/unforced-error, in/out) → cheap event + stat GT for coaching,
   and a fabrication-audit test set.

*No lane blocks on items 1–7 meanwhile — broadcast harvest + any public tennis clips carry the interim
(tennis, unlike pickleball, has an effectively unlimited public corpus). The moment owner captures
land, the in-domain finisher steps unlock (T1-2, T3-7, TS GT, spin GT).*

## I.6 Direct next steps (the first tennis wave or two, in order)

1. **T0 sport-config seam** (T0-1 tennis keypoint schema + corner-review stub fix; T0-2 wire
   `StageContext.sport` into `ball_arc_solver` and the ~12 default sites; T0-3 tennis `PhysicsParameters`
   stub). Pure code, **zero data dependency** — this is the true "can start today" work, and nothing
   tennis is trustworthy until it lands (a silent pickleball default computes the wrong geometry).
2. **T0-6 first honest tennis E2E world** on one broadcast or owner clip — everything trust-banded,
   pickleball suite still green (TM1).
3. **T1-1 ball transfer eval** — run WASB-tennis + the zoo zero-shot on tennis clips and MEASURE the
   gap to bar. This is the cheapest high-information experiment in the whole program: it tells us
   whether tennis ball tracking is nearly-free (the strategic bet) or needs the full data engine.
4. **T4-2 tennis court detector** adapt (TennisCourtDetector head) + **T4-0 owner court profiles** for
   the owner's home courts.
5. **TS-1 serve pillar kickoff** — serve segmentation + toss track + contact detection from pose;
   needs no new data, only the existing 3D-pose stack pointed at serves.
6. **T3-1 racket strung-face model spike** — decide whether the flat-rectangle PnP reuses for an oval
   strung frame or needs a new silhouette/keypoint path.
7. **Broadcast data-engine spike** — stand up the harvest→auto-label flywheel that exploits tennis
   broadcast's fixed, known camera geometry (auto-homography → auto-labels), with the broadcast-vs-phone
   domain-gap guard from PART IV rule 2.

## I.7 Definition of Done, the critical path, and demo milestones

**DEFINITION OF DONE (tennis v1).** A user with a tennis profile records a match in tennis mode (or
uploads broadcast footage) and it uploads. With zero human intervention the pipeline returns, within
≤2× rally-gated play duration: (1) a QA-passed **3D tennis world** — court+net (singles/doubles
resolved), 2 or 4 identified players, full-flight 3D ball with surface-aware bounce, both rackets in
6-DOF — where every rendered element is gate-passed or honestly trust-banded and the fusion
consistency checks hold (ball meets the strung face at contact, feet planted except on jump serves,
no penetration); (2) a **coaching card** including a serve breakdown, with ≥5 finding types tied to
jump-to 3D moments, 0 fabricated numbers, and self-relative (not pro-comparison) framing; (3) the
component gates VERIFIED **on our-camera labels, surface-stratified** (ball F1, court PCK, BODY
world-MPJPE, racket face-angle, serve contact-frame), tagged OUR-CAMERA-VERIFIED not merely
REFERENCE-VERIFIED; and (4) the launch gates cleared — an FTO opinion (SwingVision US 11,893,808 B2 /
Infinity Cube US 10,467,478) and a no-officiating-claims audit, with all NonCommercial training data
quarantined from shipped weights. **v1 is DONE when this repeats on 3 consecutive fresh matches across
at least 2 surfaces.**

**THE CRITICAL PATH (the one chain whose delay delays the tennis product):**
T0 sport seam → T1 ball transfer + 3D flight → TS serve pillar → T6 coaching (+ the FTO gate, which
runs in parallel but blocks *launch*, not development). Everything else multiplies quality/coverage in
PARALLEL but does not gate the first full-value tennis demo: T2 (body hardening — inherited bodies
already run), T4 beyond the owner's own courts (T4-0 profiles cover them immediately), multi-surface
beyond the v1 surface (PART 0 scope), T5 (scale), TL (live), T3 quality beyond swing-speed, T7 (scale).

**CAN START TODAY (zero external dependency — the true parallel front):** T0 seam (pure code) · T1-1
ball transfer eval (public tennis weights, no training) · T4-2 tennis court detector (8,841 public
labels) · the broadcast data-engine spike · TS-1 serve segmenter (pose-only, no new data) · the
match-scoring FSM (pure deterministic software — the one pillar VERIFIABLE to exactness with zero ML) ·
legal/FTO scoping. The single most decision-shaping cheap experiment is the **broadcast→phone
domain-gap probe** (T0/T1): train on broadcast GT, eval on a small our-camera held-out set, record the
drop — it decides whether broadcast is a training source or eval/physics-only.

**DEMO MILESTONES (what the owner SEES, in order — march to these):**
- **TM1 — "it runs as tennis" (days):** sport seam wired; a fresh tennis 3D world on one clip with the
  tennis court/net rendered, players placed, ball attempted, everything honestly banded, pickleball
  suite still green. [T0]
- **TM2 — "the ball and court are basically there, cheaply":** the transfer bet confirmed with numbers —
  our-camera ball F1 clears bar with ~10× less data than pickleball; tennis court PCK to bar. [T1-1/T1-2,
  T4-0/T4-2]
- **TM3 — "it sees the serve":** the first serve card — toss, contact height, kinematic-sequence order,
  serve speed vs radar, foot-fault advisory — every number gate-passed or trust-banded. [TS-1..TS-4]
- **TM4 — "one world, spin-aware":** a fused tennis world where the ball meets the racket at contact,
  spin sign agrees with bounce behavior, and surface is classified and its physics applied. [T1-4/T1-5,
  T4 multi-surface, TF-1/TF-2]
- **TM5 — "it coaches my tennis, and a friend can use it":** a fabrication-audited tennis coaching card
  (incl. serve breakdown) + a friend onboarding into tennis mode. [T6, T7 partial]

# PART II — TENNIS RESEARCH VERDICTS

> Scope note: every metric below is the source's own number on the source's own split; "our-camera" means a single courtside iPhone, the product's actual capture. **VERIFIED=0 for every tennis pillar** until a pre-registered gate passes on real our-camera labels — the abundant tennis ground truth is overwhelmingly broadcast/multi-camera domain, and the pickleball in-domain-data lesson (measured 4×) still binds. Where a claim is unverifiable or a license is unclear, it is flagged.

---

## Ball 2D tracking

**(a) Carries over.** Tennis is the birthplace of this entire line of work — TrackNet (Huang 2019) was built on tennis broadcast — so the pillar transfers with near-zero re-architecture. The WASB-SBDT default stays unchanged (3-frame heatmap, HRNet backbone, position-aware training + online temporal-consistency), the `blurball` training fork stays and becomes *more* valuable (tennis blur is worse), the TrackNetV3 checkout and union-candidate-pool design stay, and the whole eval harness (`benchmark_ball_trackers.py`, `ball_benchmark.py` with hit-radius / teleport / max-jump-gap gating) plus the pre-registered held-out ledger carry over. Top-K sidecar emission, physics-fill, and the measured/hidden/predicted/low-confidence trust bands all transfer. Kill-list discipline (CVAT-only fine-tunes that regress held-out, veto-fusion that collapses recall, VNDetectTrajectories rung-1) stays rejected.

This inverts pickleball's situation: pickleball hit a measured zero-shot wall (best held-out ball F1 ~0.6969 vs a 0.7248 bar) because public pickleball detection data was exhausted and no checkpoint had ever seen a holey plastic ball. Tennis is *in-domain* for the zoo, so transfer costs dramatically less in-domain data — roughly an order of magnitude less than pickleball needed.

**(b) Must change / build.**
- **Swap checkpoints, not architecture:** load upstream tennis WASB/TrackNet weights (both zoos ship tennis checkpoints in `MODEL_ZOO.md`) instead of pickleball fine-tunes. *(small)*
- **Re-derive teleport / max-jump gates for serve speed:** current 160 px/frame teleport + 3-frame max gap will reject *every real serve*. A 120–150 mph serve = 53–67 m/s moves ~0.9–1.1 m/frame at 60 fps (~13 ball-diameters), which in 1280×720 can exceed a large fraction of frame width. *(medium)*
- **Make 120–240 fps the default for serve-containing rallies**, not an option — the 3-consecutive-frame trackers lose a ball that jumps a large fraction of the frame between samples. *(medium)*
- **Promote motion-blur streak modeling to first-class** (BlurBall-style joint position + blur orientation θ + half-length ℓ): at serve speed the ball is a streak, and the blur length is a free per-frame velocity/direction cue. *(large)*
- **Collect a modest amateur low-angle iPhone fine-tune set** (hundreds–low-thousands of frames) + SST; all tennis SOTA is broadcast high-angle, our view is still OOD. *(medium)*
- **Tiled/high-res inference for the far half:** the 78 ft singles court puts a far-baseline ball at a few pixels, below the small-object floor of a downsampled heatmap. *(medium)*
- **Retire TrackNetV4 from the live path** (repo already pins it "blocked-no-usable-weights"; independent V5 reproduction shows V4 tennis F1 *below* V2). *(trivial)*

**(c) SOTA.**
- **ADOPT — WASB-SBDT** (MIT, github.com/nttcom/WASB-SBDT). BMVC 2023; AP beats prior SBDT 7.8–16.8%, multi-sport mean F1 ~88.2, 1.5M params. License is **confirmed MIT** (correction applied — earlier "verify" is closed).
- **ADOPT — TrackNetV2** (open, community forks). Tennis F1 reported **0.9037 in the WASB paper up to 0.9677 in the V5 reproduction** (1280×720, 4 px tolerance). Pin the split + tolerance in our own gate to avoid cherry-picking.
- **ADOPT — TrackNetV3** (open, github.com/qaz812345/TrackNetV3). ACM MMAsia 2023. *Correction:* the ~0.94 precision / 0.80–0.83 recall figures are **RacketVision reproductions** (implied F1 only ~0.86–0.88); the 0.90–0.97 F1 numbers come from the **separate native TrackNetV2 tennis test set** — keep these two eval regimes explicitly distinct. MS-TrackNetV3 on RacketVision reaches **MDE 1.96 px** (corrected from 1.70; P 0.945 / R 0.880 / mAP 81.9).
- **SKIP — TrackNetV4** (MIT, TF). ICASSP 2025. In-repo weights undeserializable; V5 repro shows V4 tennis F1 0.9581 **below** V2's 0.9677.
- **WATCH — TrackNetV5.** arXiv 2512.02789; headline tennis F1 0.9859 / 114 FPS on T4. *Correction:* not "no code" — an **official proprietary all-rights-reserved SDK** exists (github.com/codelancera-offical/TrackNetV5-SDK, Shanghai Code Zero) with training+inference code, but **weights and datasets are withheld**, so it remains **unreproducible / watch-only**. SDK may postdate the Jan-2026 snapshot.
- **ADOPT — BlurBall** (dataset public; model license unstated — verify repo). arXiv 2509.18387. F1 97.17 vs WASB 95.58 on a 64,119-frame set (62% blurred); **blur-length MAE 1.2 px vs WASB 3.1 px**; trajectory error 84.4→53.0 px with blur fitting.
- **WATCH — YOLO-Ball** (license not stated). P 82.2%, mAP@0.5 70.9% on tennis; weaker recall than heatmap SOTA.

**(d) Sources.** arXiv 1907.03698, 2311.05237, 2509.18387, 2512.02789, 2409.14543, 2511.17045; github.com/nttcom/WASB-SBDT (+ MODEL_ZOO.md); github.com/qaz812345/TrackNetV3; gitlab.nol.cs.nycu.edu.tw TrackNet.

---

## Ball 3D flight / physics / spin

**(a) Carries over.** The RK4 drag+gravity+Magnus ODE core (`_rk4_step_with_magnus`, `flight_simulator.py`; `ball_arc_solver.PhysicsParameters`) transfers verbatim — the force model FD=½·Cd·ρ·A·v², FL=½·Cl·ρ·A·v², Fg=mg is identical physics. The event-anchored per-segment BVP arc solver with Huber reprojection + LOO residual gating is sport-agnostic. The synthetic-lift training pattern (`generate_trajectory_pair`: physics sim + detector-noise injection → 2D→3D uplift net) is exactly the TT4D/UpliftingTT recipe and is needed because **tennis, like pickleball, has zero real 3D ball labels**. Spin is already represented as (axis, scalar) with Magnus lift_dir = axis × v̂, so tilted kick/slice axes are already expressible — only the magnitude range and Cl(S) slope retune. The size-depth residual, monocular + iPhone-sidecar assumptions, trust bands, and net-clearance scaffolding all reuse.

Critically, the ball-3D schema **already reserves a per-frame `spin_rpm` field** (`solve_ball_arcs.py`) — but it is **hardcoded `None`**. Spin is therefore net-new, not a port.

**(b) Must change / build.**
- **Reseed geometry:** diameter 0.0742→~0.0657 m, mass 0.0255→~0.0580 kg; make ρ_air altitude-aware (Madrid/Denver clay ~15% lower ρ flattens arcs). *(trivial)*
- **Tennis drag:** replace Cd=0.33/0.45 with a **banded, Reynolds/wear-dependent Cd**. Seed ~0.507 (Goodwill/Cross new-ball mean) but treat as a per-clip tunable prior — there is a **genuine literature disagreement** (Mehta reports 0.6–0.7 for new balls). *Correction:* the fuzz effect is ~**10%** of Cd and worn-ball change ~**6%** (Goodwill/Chin/Haake) — **drop the earlier "up to ~40%" and "toward ~0.4" figures**; keep only the qualitative "fuzz makes Cd Re/wear-dependent and contested." *(small)*
- **Stronger Magnus:** change STEYN_CL_PER_SPIN=0.195 to a tennis Cl(S)≈0.5–0.6·S with a small nonzero-spin intercept (~0.05–0.1); widen sampled spin to S up to ~0.45 (3200–5500 rpm topspin/kick/slice). *(small)*
- **Raise speed corridors — correctness bug, not tuning:** `selection_max_speed_mps=35` and `max_plausible_speed_mps=35` silently reject serves at 54–67 m/s; synthetic launch ranges (4.5–20 m/s) never generate them. *(small)*
- **Surface-conditioned, spin-coupled bounce:** replace restitution=0.58/friction=0.16 with per-surface {COR e_y, friction μ, spin-coupling}; delete pickleball two-bounce logic. *(large — see Multi-surface, Bounce.)*
- **Serve-speed + spin-axis as first-class outputs** — monocularly recoverable *because* drag (~5g at serve speed) makes the arc strongly curved/observable. *(medium)*
- **Spin: ship CLASS first (topspin/backspin/slice/flat), then a coarse 3-band RPM.** Precise RPM (±5%) is feasible only for serves/clear arcs at 100–240 fps; full 3D spin-axis is **not** reliable monocular (even a specialized event camera errs ~33° on axis). *(large / research-open)*

**Honest ceiling on spin:** the load-bearing "97% spin accuracy" (UpliftingTableTennis, WACV 2026) is **binary topspin-vs-backspin on a synthetic set (97.1%), falling to 89.5% on real-world captured footage (TTHQ)** — *correction:* TTHQ is captured, **not broadcast**; the broadcast result (92.0%) is the *separate* SPT paper, do not conflate. No published RPM-magnitude error, table-tennis geometry only. Direct rotational felt/logo tracking blurs out above ~4500 rpm on a phone — kick serves (~5000 rpm) sit at that wall.

**(c) SOTA.**
- **ADOPT (pattern) — UpliftingTableTennis** (GPL-3.0 code + weights + TTHQ). WACV 2026. *Correction:* authors are **Kienzle, Ludwig, Lorenz, Satoh, Lienhart — not "Blank et al."** GPL-3.0 contaminates a closed product: reimplement clean-room, do not ship its weights.
- **ADOPT (physics) — Goodwill/Cross wind-tunnel tables + Cross oblique-bounce model** (published facts, freely usable as constants). Cd mean 0.507±0.024 (range 0.453–0.567), Re 8.5e4–2.5e5; ITF COR ≈ grass 0.73 / clay 0.83 / hard 0.80 (**flag as approximate — no single ITF primary doc verified these exact per-surface numbers**).
- **ADOPT — WASB** (confirmed MIT) as the 2D front-end every trajectory-inverse spin method consumes.
- **ADOPT — "Where Is The Ball"** (3D trajectory from 2D monocular). **CVPR 2025 Workshop (CVSPORTS)** — not just a preprint. Tennis (Real TrackNet dataset): **87.21% landing accuracy, F1 0.807, landing error 0.63 m** vs SynthNet 3.58 m; robust to ±25 px noise. Code "to be released" — availability unverified.
- **ADOPT — TT4D** (CC BY 4.0). **CVPR 2025**. Racket pose at impact recovered to only **26.4±4.4° orientation error** vs mocap — this **bounds ball-impact inversion**: pure inversion cannot give a trustworthy face angle, stays advisory. Dataset **140+ hours** (corrected from "146h"), 211,534 reconstructed points.
- **WATCH — TT3D** (CVPR 2025, IEEE Xplore); **SPT** (arXiv 2504.19863, 92.0% topspin/backspin on real broadcast, F1 0.917 — license/code unverified); **single-camera TSSE+PSE** (Measurement 2026, paywalled, speed MAE 4.81% / spin RMSE 3.42% *repeatability, not radar-validated absolute*).
- **SKIP — event-camera spin** (Gossard, CVPRW 2024): axis MAE 32.9±38.2°, needs event hardware + visible logo lines — not a consumer phone. **SKIP — SwingVision/Hawk-Eye ELC** as components (proprietary; SwingVision proves single-cam tennis 3D is commercially viable, Hawk-Eye sets the multi-camera accuracy bar).

**(d) Sources.** Mehta 2001/2008; Goodwill 2004; Cross (Tennis Warehouse aerodynamics2 / kickserve / bounce PUBLICATIONS/41,52); Sakurai 2013 (kick ~3215 rpm); arXiv 2511.20250, 2504.10035, 2506.05763, 2605.01234, 2504.19863, 2404.09870; github.com/KieDani/UpliftingTableTennis; `solve_ball_arcs.py` (spin_rpm=None stub).

---

## Bounce-contact / line-calls

**(a) Carries over.** This pillar ports best of the perception pillars because the codebase is *already multi-sport at the deciding point*: `court_templates.py` ships a validated tennis template, and `Sport = Literal["pickleball","tennis"]`. The whole `sigma_bounce` camera-geometry model (`ball_inout_uncertainty.py`: sqrt of reproj² + depth² + ballradius² + localization², self-calibrated 6-DOF pose from the 4 manual corners, grazing-angle diagnostic) ports wholesale — only one constant is pickleball-specific. `classify_ball_line_calls` (`ball_line_calls.py`) with per-call uncertainty_radius_m and `too_close_to_call` runs on the tennis template unchanged for baseline/sideline calls. `event_fusion.py` (audio onset + ball inflection + wrist-velocity → contact window, with trust bands) and the `audio_onsets_v2` HFC/spectral front-end carry over as pipelines. `shot_taxonomy._has_excess_bounce` **is effectively the tennis double-bounce-loses-point rule** and transfers directly. Governance (advisory-not-officiating, `too_close_to_call`) carries — and is now table stakes.

**(b) Must change / build.**
- **Swap the one pickleball constant:** replace `PICKLEBALL_DROP_TEST_HEIGHT_M` (1.9812 m) with the ITF tennis rebound spec (254 cm drop → 135–147 cm rebound) to recompute `v_z_ref`; higher tennis impact speeds physically enlarge sigma_depth. *(trivial)*
- **Framerate-gate in/out:** at 30 fps a 54–67 m/s serve moves ~1.8–2.2 m/frame, so the ±2-frame `BOUNCE_DETECTION_FRAME_WINDOW` inflates sigma_depth to *meters*. Require ≥120–240 fps for confident serve calls; make the window adaptive to measured speed. **This is the dominant tennis failure mode.** *(medium)*
- **New call types:** service-box in/out (fault), singles-vs-doubles sideline switching by point context, net-cord lets, foot faults; two-bounce → one-bounce. *(medium)*
- **Retune audio:** a strung-racket impact is a ~5 ms "thwock" (vs table-tennis ~1.3–1.8 ms, vs a pickleball "pock") — re-fit bandpass, min-separation, HFC thresholds; add hit-surface classification (racket/ground/net-cord). *(medium)*
- **Surface-aware restitution/friction + clay ball-mark detection** as a *new* mark-anchored review capability (zero pickleball analog). *(large / research-open)*

**Officiating context (corrections applied).** **Roland Garros did NOT adopt ELC in 2025** — it was the lone Slam still using human line judges; the correct 2025 milestone is the **ATP Tour mandating ELC at all events, including clay (Madrid, Rome)**. **Foxtenn is the pioneering clay mark-verification system, not the sole tour-approved one** — ATP uses Hawk-Eye Live on clay. Hawk-Eye's documented figure is the **3.6 mm advertised average** (the "~2.2 mm" has no primary source — drop it); camera count is **~6–12 per court (up to ~18 in some ELC Live setups)**, not "10–18." "Where Is The Ball" 87.21% is **landing accuracy on the Real TrackNet dataset**, not an average across datasets.

**(c) SOTA.**
- **SKIP (reference only) — Hawk-Eye/ELC** (Sony, proprietary). 3.6 mm advertised; ~6–12 cameras @340 fps; ~1 s to call. **SKIP — Foxtenn Real Bounce** (proprietary; clay "real image of the bounce").
- **WATCH — SwingVision** (proprietary). Vendor: 97% within 10 cm single-camera, 99% two-camera; officiated 500+ USTA matches 2024 — the realistic single-cam bar.
- **ADOPT — "Where Is The Ball"** (CVPRW 2025; check repo license). Validates synth-lift for tennis at 87.21% landing accuracy.
- **ADOPT — TTNet / OpenTTGames** (dataset research; GPLv3 code). Real-time 120 fps bounce/net event spotting — reference design for a *learned* event head (roadmap P1-6) to augment heuristic cusp+gap anchors.
- **ADOPT — TrackNetV3/V4 (tennis ball baselines); ML acoustic hit/bounce timing** (offline >95%, on-court ~85%; tennis contact ~5 ms — the audio retune target).
- **WATCH — Monocular ELC of Tennis** (arXiv 2107.09255) — single-camera method + honest error framing.

**(d) Sources.** ITF Rules of Tennis; arXiv 2107.09255, 2506.05763, 2004.09927, 2504.10035; CVPRW 2025 "Where Is The Ball"; github.com/HaydenFaulkner/Tennis; Foxtenn / tennis.com clay-ELC coverage; internal `ball_inout_uncertainty.py`, `ball_line_calls.py`, `event_fusion.py`.

---

## Body / 3D pose

**(a) Carries over.** The frozen backbone transfers at the model level — **SAM-3D-Body + MHR70** is a generic anatomical rig (MHR Apache-2.0), a tennis player is a human. The classical world-grounding chain (person-masked LK+MAD camera tracking, foot-plane anchoring, MAD+Gaussian / MHR-latent smoothing — the SMART/WorldPose-winning architecture) is sport-agnostic. Person track + ReID (YOLO26m + BoT-SORT/OSNet), MHR-latent temporal smoothing (P2-2), the challenger benchmark-only protocol (P2-7), and SMPL/SMPL-X interop via SOMA-X (needed to consume SMPL-based tennis GT) all reuse. Wrist-3D extraction extends directly to serve-contact features.

**Genuinely good news:** unlike pickleball (BODY *never* GT-validated), tennis has real 3D benchmarks — **CalTennis** and **AthletePose3D** — so the pillar can finally earn its *first* VERIFIED body gate.

**(b) Must change / build.**
- **Phase-gate OFF foot-lock during serve flight** (both feet airborne at full extension) — the single most load-bearing world-grounding trick will otherwise *drag the airborne body down*. Highest-severity grounding bug tennis introduces. *(medium)*
- **Per-phase adaptive MHR-latent smoothing:** near-zero window across the serve acceleration snap (shoulder IR ~2400°/s, wrist ~1900°/s), W=9 elsewhere — the documented SMART over-smoothing failure, amplified. Make the wrist-latent hybrid the default. *(medium)*
- **Axial limb rotation (humeral IR/ER, forearm pron/sup) as a separately-trust-banded output** — shoulder IR contributes ~41% and wrist ~32% of racket velocity, and it is exactly the DOF monocular HMR is weakest at. Likely UNVERIFIED until racket-6DOF fusion lands; **do not ship as measured.** *(research-open)*
- **Racket-aware occlusion** (PromptHMR mask/box prompts) and a **two-handed-backhand hand-hand contact model.** *(medium each)*
- **Court-anchored scale/translation** (singles 8.23×23.77 m): monocular translation error is 0.9–3.6 m at 13–17 m capture distance (CalTennis) — lean on court plane + sidecar ARKit/LiDAR. *(small)*
- **8-stage serve-phase segmenter** as the gating spine + stance priors (open/closed/neutral, lunge, split-step). *(large / medium)*
- **GT-validate on CalTennis + AthletePose3D fine-tune** — the highest-leverage tennis body task. *(large)*

**(c) SOTA.**
- **ADOPT — SAM 3D Body + MHR** (SAM License for 3DB, permissive w/ field restrictions; MHR Apache-2.0). SOTA across 3DPW/EMDB/RICH/Harmony4D/COCO/LSPET — but on general in-the-wild sets, **not tennis serve dynamics.**
- **ADOPT — Fast SAM 3D Body** (follows SAM-3D-Body license; community repo). *Correction:* it is a **training-free acceleration framework, up to ~10.9× end-to-end** (not a "distilled substitute"); the "accuracy-preserved" claim is contested on inspection — re-verify checkpoint parity for tennis before trusting the speedup in an L3 gate.
- **WATCH — PromptHMR** (research license — verify). CVPR 2025. **Best translation on CalTennis: 0.942 m** (vs WHAM 2.664 / GVHMR 3.587 / TRAM 2.340 / GENMO 2.560); PA-MPJPE 84 mm.
- **WATCH — KASportsFormer** (research). SportsPose 58.0 mm / WorldPose 34.3 mm MPJPE; targets the "critical motion finishes in a moment" problem — but skeleton-only, no mesh.
- **SKIP — GVHMR** (research). CalTennis translation 3.587 m (worst of 5); our P2-7 spike found its gravity premise did not help tripod clips.
- **WATCH — WHAM / TRAM / GENMO / Human3R** (per-repo research licenses). All materially worse than PromptHMR on tennis translation.

**Corrections applied.** **CalTennis is CC-BY-NC-4.0 (NonCommercial)** on HuggingFace (only the *paper* is CC-BY 4.0) — **not commercially usable as-is**; it provides **multi-view-triangulated 3D GT (a lower bound), not documented SMPL annotations**, and serve-specific labeling is **UNCONFIRMED**. **AthletePose3D is general athletic (12 sports, ~1.3M frames, optical mocap, non-commercial), not tennis-specific** — the 214→65 mm fine-tune gain is on its own mixed-sports data, so the "VERIFIED on tennis motion" claim rests primarily on **CalTennis**.

**(d) Sources.** arXiv 2602.15989, 2603.15603, 2606.20542 (CalTennis), 2503.07499 (AthletePose3D), 2507.20763, 2504.06397, 2403.17346, 2510.03921; github facebookresearch/sam-3d-body, MHR, NVlabs/SOMA-X; PMC serve-kinematics + 8-stage-serve refs.

---

## SERVE biomechanics (NEW pillar)

**(a) Carries over.** The 3-stage coaching pipeline (deterministic features → non-LLM rule/reference-range comparator → format-locked LLM) maps 1:1 onto serve coaching; "serve" is already a `SHOT_TYPE` in `shot_taxonomy.py`, and `contact_windows.py` already fuses wrist-velocity + ball-inflection + audio into a contact instant with trust bands — exactly the event the 8-stage timeline anchors to. Foot-contact phases, court/net calibration (for foot-fault), the **iOS sidecar's already-shipped 240 fps + ARKit pose + gravity + LiDAR + locked exposure** (the single most important enabler), the ball zoo + arc solver (for toss/serve-speed), and the racket-6DOF scaffold all reuse. Trust-band / ledger discipline is *essential* here given injury-claim liability.

**(b) Must change / build.**
- **Serve reference-range library** (signed versioned JSON) seeded from **Kovacs–Ellenbecker 8-stage model + Elliott ITF corpus**: max external rotation 172±12° reached ~0.09 s pre-contact, shoulder abduction ~110° at contact, front-knee flexion 24±14° at contact, trunk tilt 48±7°, contact racket velocity 38–47 m/s, peak vertical GRF 1.68–2.12 BW, legs+trunk = 51–55% of energy. Racket-head velocity = **54.2% shoulder internal rotation + 31.0% wrist flexion + 12.9% horiz flexion/abduction** (Elliott/Reid). Each range signed with a citation + owner-review flag. *(medium)*
- **8-stage temporal segmenter** (start, toss-release, loading, cocking/trophy, acceleration, contact, deceleration, finish). *(medium)*
- **Mandate 240 fps; refuse/heavily-band 30–60 fps serves:** the acceleration phase (max-ER→contact) lasts **<0.01 s** and racket velocity is 38–47 m/s — 60 fps (16.7 ms/frame) cannot sample peak angular velocities or contact speed at all. *(small)*
- **Declare distal velocity (shoulder IR ~42 rad/s ≈ 2420°/s, wrist ~34 rad/s ≈ 1950°/s) and racket-head-speed trust-banded / out-of-scope** until racket-6DOF retool + a validated gate — the platform's most-wanted serve number is its least reliable markerless signal. *(small)*
- **Racket retool** flat-rectangle → strung elliptical long lever (string-bed plane + lever offset); **ball-toss tracking** (zenith:impact-height ~1:1.5); **radar-calibrated serve-speed** (degrade above ~100 mph); **precision-first foot-fault advisory**; **injury markers as non-clinical kinematic proxies only** (video cannot measure ~0.5–0.75 BW shoulder distraction or ~300 N·m torques — medical-liability). *(medium–large)*

**(c) SOTA.**
- **ADOPT — Kovacs–Ellenbecker 8-stage model** (Sports Health 2011; numeric norms are free facts). USTA-adopted.
- **ADOPT — OpenCap** (Apache-2.0, free). Validated RMSE 2.0–10.2° vs marker-based — **adequate for proximal segments (pelvis/trunk/knee), NOT distal shoulder/wrist velocity.**
- **ADOPT — monocular serve-speed vs radar** (TAMU thesis + MIT-style reimpls). "Closely aligns with radar"; degrades >100 mph. **ADOPT — HRNet markerless mechanical-work** (J Sports Sci 2025): serve external work differed from marker-based by only 7.3 J / 9.3 J — whole-body energetics *are* recoverable markerless even when distal velocities are not.
- **WATCH — Talking Tennis** (CC-BY paper, no code; THETIS-based). **SKIP — Theia3D** (proprietary; ~6.8–9.1° RMSD yardstick only).

**(d) Sources.** PMC3445225 (8-stage); Elliott/Reid ITF biomech; Xsens serve kinematics (PMC11504545); Sakurai 2013; frontiersin fspor 1451174 (racket velocity); OpenCap S0021929024002781; arXiv 2510.03921; TAMU serve-speed thesis; ITF Coaching & Sport Science Review.

---

## Stroke classification

**(a) Carries over.** SAM-3D-Body skeletons + arc-solver trajectory + fused contact frame are exactly the two inputs racket-sport SOTA fuses. The 3-stage grounded-LLM pattern is directly validated by "Talking Tennis." Abstain-below-confidence gating (`min_shot_type_confidence`, `shot_type_abstained`), the per-shot record schema, WiLoR hand crops (for handedness / 1H-vs-2H), and the in-domain + CVAT + SST unlock all transfer.

**The #1 architectural delta:** pickleball's `_classify_shot_type` derives shot TYPE **purely from ball-arc geometry** (launch/landing/peak/speed/kitchen tests, zero pose). That works because dink/drop/drive/lob/smash *are* trajectory buckets. It **physically cannot** recognize forehand vs backhand, 1H vs 2H backhand, or topspin vs slice — those are **pose/swing phenomena**, so tennis needs a NEW pose+trajectory recognition pillar.

**(b) Must change / build.**
- **Add a pose+trajectory stroke-recognition pillar** (port the BST fusion pattern); keep arc-geometry only for coarse context. *(large)*
- **Replace the SHOT_TYPES enum + kitchen/NVZ/erne/atp/tweener logic** with tennis: serve(flat/kick/slice), FH & BH (topspin/slice/flat), 1H/2H backhand, volley, half-volley, overhead, drop, lob, approach, return; contact_zone → baseline/no-mans-land/service-box/net. *(medium)*
- **Swing-phase segmentation** anchored on the fused contact frame + wrist-velocity zero-crossings. *(medium)*
- **Infer topspin vs slice from swing path (low-to-high vs high-to-low) + bounce**, not measured spin (spin period is sub-frame at 30–60 fps); 240 fps helps. *(large)*
- **Handle serve-class rarity / imbalance** — gate on **macro-F1**, not top-1. *(small)*

**(c) SOTA.**
- **ADOPT — BST (Badminton Stroke-type Transformer)** (PyTorch, verify repo license). *Corrections:* the headline BST-CG-AP (25 merged classes) is **83.22% top-1 / 0.8097 macro-F1 / 95.94% top-2**, **+0.87% over TemPose-TF, +5.64% over ST-GCN** (fix the 35-class 76.95%/+1.2%/+4.1% figures to whichever table you cite); arXiv Feb 2025, **accepted CVPRW 2026**. Authors state ball-trajectory fusion "is likely a trend for racket sports."
- **ADOPT — PoseC3D** (Apache-2.0, MMAction2). NTU60-XSub ~94.1%; documented robustness to pose noise vs GCNs — the recommended skeleton backbone.
- **WATCH — Talking Tennis** (arXiv, no code); **DG-STGCN** (Apache-2.0; tennis-customized variant unreleased/unverified); **InternVideo2 / VideoMAE V2** (**non-commercial** — VideoMAEv2 CC-BY-NC, InternVideo2 restricted; verify before product use).
- **SKIP — SwingVision** (proprietary; ICC=0.97 stroke detection). *Corrections:* the same study reports **spin recognition ICC=0.76** (so SwingVision *does* attempt spin — nuance your "spin unmeasurable at phone fps" claim), n=5 elite-junior; the "confuses volleys with flat forehands / swaps FH-BH" claim is **anecdotal user report, not a measured error rate.** **SKIP — event-camera spin** (needs hardware).

**Corrections applied.** THETIS's 55 subjects are **31 beginners + 24 experts** (not all amateur); **79.17% is the recent Talking-Tennis/monocular figure — older THETIS studies report 86–95%** using depth/skeleton streams, so 79% is not an absolute ceiling.

**(d) Sources.** github/THETIS-dataset; arXiv 2502.21085v2 (BST), 2104.13586 (PoseC3D), 2210.05895, 2403.15377, 2510.03921; TenniSet (Faulkner & Dick); mdpi 2076-3417/13/10/6195 (SwingVision validity); mmaction2 skeleton zoo.

---

## Racket 6-DOF

**(a) Carries over.** The pillar transfers at the *architecture* level almost unchanged: the `racket_pose_estimate.json` contract, `virtual_world` consumer, and provenance bands need no schema change. **IPPE planar-PnP still holds** — a strung face is a rigid planar ellipse with a well-defined normal. `paddle_pose_fused` end-to-end (MHR70 hand-frame H(t), constant grip-transform G via weighted Wahba `_solve_wahba`, ball-reflection face-normal evidence, wrist-gated box correction, per-segment 1-DOF grip-roll IoU search) reuses. The SLERP one-euro rotation smoothing (proxy jitter 23–53°→5°/frame) carries — and racket needs it *more* (longer lever amplifies rotational jitter into tip jitter). Our 5-keypoint schema ≈ RacketVision top/bottom/handle/left/right. `_paddle_footprint_local` already models the face as an **ellipse**, not a rectangle.

**(b) Must change / build.**
- **Geometry constants:** 16"×8" rect / 5.25" handle → **27" (0.686 m) racket, elliptical head ~318×265 mm, long throat+grip.** *(small)*
- **Longer lever t_g (~3–4×):** small H(t) rotational noise → large face-tip excursion; re-tune closed-form t_g least-squares, make lever explicit. *(medium)*
- **Tennis grip-prior library** (Eastern/Semi-Western/Western/Continental/serve-pronation) replacing the single continental-paddle R_g; **per-segment-constant-G is weaker** (serve pronation spins the face fast within one contact). *(medium)*
- **Two-handed backhand:** replace single hand-side selection with a **dual-wrist grip model** (both wrists constrain one racket frame). *(large)*
- **Swing-speed via lever arm:** v_head = v_wrist + ω×r_lever (racket head 30–45 m/s at serve, far above wrist speed — wrist displacement badly under-reads). *(medium)*
- **Contact-point-on-string-bed output + serve stroke class**; possible **blur-axis-as-orientation channel** for the near-transparent strung face on fast swings. *(medium / research-open)*

**(c) SOTA.**
- **ADOPT — RacketVision (RTMDet-M + RTMPose-M)** (**MIT** code/data/weights on HuggingFace). AAAI 2026 Oral. Tennis: 150,399 frames / 7,395 racket annos; RTMPose PCK@0.2 **89.6%**, MPJPE 5.34 px, detection mAP@0.5 79.4%. **Face-WIDTH (left/right) keypoints are the weak axis** — the exact axis that sets face-normal roll. *Correction:* the multi-sport table reports **Left 79.7% / Right 80.1%** — verify the cited "64.85%" floor against its specific config; state face-width PCK as ~80% and keep the >92% structural contrast (solidly confirmed).
- **ADOPT — RTMPose/RTMDet** (Apache-2.0, MMPose).
- **ADOPT (bound) — TT4D** (CC BY 4.0). Ball-impact inversion face angle bounded at **26.4±4.4°** → stays advisory. Dataset **140+ hours**.
- **WATCH — BlurBall** (license unstated); **CalTennis** (CC-BY-NC-4.0). **SKIP — TT-robot 6D pose** (not open; uses depth, not our monocular case).

**Sensor rackets (corrections).** Do **not** cite "Babolat Play ~99.3% / Zepp ~96.7%." The primary validation studies report **Zepp racket-speed ICC=0.983**, stroke-classification **kappa Babolat 0.730 / Zepp 0.612**, and **poor impact-location agreement (Zepp κw=0.217, Babolat κw~0.412 — verify)**. Hardware is largely **discontinued by 2020–2021** — treat as calibration, not officiating, and sourcing-hostile.

**(d) Sources.** arXiv 2511.17045 (RacketVision), 2605.01234 (TT4D), 2606.20542 (CalTennis), 2509.18387 (BlurBall); github.com/OrcustD/RacketVision, open-mmlab/mmpose; sensor-racket validation (Fuss Sports Biomech 2018, IJCSS 2019, PMC8321100).

---

## Court / net calibration

**(a) Carries over.** Tennis is where this pillar flips from data-starved to data-rich. `court_templates.py` **already ships a correct tennis `CourtTemplate`** (78×36 ft, singles 27 ft, net 42"/36" center=0.914 m posts=1.067 m, service line 21 ft, alleys) and `line_segments_m` emits service lines — the metric skeleton and world frame carry over almost as-is. `court_zones.py` already builds the four service boxes + doubles alleys. The whole calibration back-end reuses: `court_calibration_metric15.py` (planar homography + focal grid-search + solvePnP), the LM point+line residual seam (independently validated by TVCalib/PnLCalib), the ChArUco k1/k2 protocol, PoseGravity/AnyCalib/GeoCalib, `CourtProfile` registry, and the heatmap-keypoint → 4-point-homography → reference-snap algorithm pattern. The `OVERLAPPING_COURT_CALIBRATION_GOAL.md` stack (HSV mask, net-crop, Hough clustering, shadow-removal fallback) is essentially a re-implementation of the amateur-court paper and is already tennis-shaped. Held-out ledger + CVAT owner-labeling reuse; only the keypoint name set and reference template change.

**Rehabilitated architecture:** tennis's abundant data **rehabilitates the exact heatmap-then-points CNN our own `court_unet_v2` KILLED** (ledger rows 70–72: PCK@5 0.017–0.056 vs a 0.95 gate). The failure was pickleball data scarcity, not the architecture.

**(b) Must change / build.**
- **Tennis 14–16+ keypoint schema** built around service boxes (not the kitchen): 4 doubles corners, 4 singles-baseline corners, 4 service-box outer corners, 2 center-service-line "T" junctions, 2 baseline center marks, + net-top points (posts/center, optional singles-stick tops). Extra coplanar points *improve* single-view PnP identifiability. *(medium)*
- **Singles/doubles court-mode flag + runtime classifier:** the same lines serve both formats, so format cannot be read off geometry — fuse person-count (2 vs 4), lateral standing positions, serve-landing box. A wrong `court_mode` silently corrupts in/out, zones, AND the net model. *(medium / large)*
- **Net model:** generalize `net_top_height_m_at_x` from a single linear post→center ramp to a **piecewise/parametric profile** — center strap pinned to 0.914 m, rising to 1.07 m at doubles posts OR at **singles sticks (±(27/2+3) ft) with the outboard alley sagging below** (a genuine double-dip the linear model cannot represent). *(medium)*
- **Multi-surface line handling:** surface classifier + court-color-context filter (the 7×7 "neighbors match court color but pixel does not" rule from the 2024 amateur paper) + shadow preprocessing. Clay off-white-on-red, worn grass, and hard-court shadows are the low-contrast failures. *(large)*
- **Do NOT ship public weights as-is** — reimplement the architecture under our own license, pre-train on broadcast, fine-tune on our-camera. *(large)*
- **Re-anchor the bar:** Hawk-Eye's 2.2–3.6 mm is a 6–10-camera number, physically unreachable single-cam — set a **pixel/metric-ft residual target** and keep all calls advisory. *(trivial)*
- **Capture (see Multi-surface):** default tennis to **4K60** (far-baseline ball ~4 px @1080p → ~8 px @4K at ~24 m), elevated behind-baseline mount, surface + sport tag in the sidecar. Downgrade LiDAR/ARKit-plane grounding: LiDAR range ~5 m ≪ 24 m court, so rest on **line-homography**, not the ARKit floor plane.

**(c) SOTA.**
- **WATCH — yastrebksv/TennisCourtDetector** (**NO license file → all-rights-reserved**; usable for research/seed, **not shippable**). TrackNet-like, 14 kpt + center, 8,841 imgs (hard/clay/grass). *Corrections:* the **0.963 precision / 0.961 accuracy / 1.83 px median is the FULL post-processing config** (base + kp-refine + homography); the **raw base model is 0.936 / 0.933 / 2.83 px** — do not present 1.83 px as raw detector output.
- **ADOPT (method) — Accurate Tennis Court Line Detection on Amateur Recorded Matches** (arXiv 2404.06977, no clean-licensed code). *Correction:* authors are **Sameer Agrawal, Ragoth Sundararajan, Vishak Sagar — NOT "Gao/Farrukh" or "Gao/Cai."** Directly targets our low-angle/shadow/worn-court regime.
- **WATCH — PnLCalib** (code **GPL-2.0** — viral, method-only) and **TVCalib** (MIT *asserted, unverified this pass*). SoccerNet-Calibration SOTA family; point+line joint optimization transfers. *Correction:* SoccerNet-Calibration-2023 is **~22,816 images**, not 25,506.
- **ADOPT (temporal) — Farin RANSAC line + iterative court-model tracking** (~6 ms/frame; classical, freely reimplementable).
- **SKIP — Automated Tennis Player/Ball/Court (arXiv 2511.04126).** *Correction:* it reports **~3.8 px avg keypoint error (NOT 1.83 px), uses a custom ResNet50, and does NOT reuse yastrebksv** — the "identical → reuses that detector" flag is **false and removed.** Systems-integration reference only.

**Dataset license trap (correction).** The **Gholamreza HuggingFace dataset's "MIT" tag does NOT confer commercial rights** — it is an admitted re-upload of the same all-rights-reserved broadcast frames from yastrebksv's repo. A third-party MIT tag on someone else's copyrighted broadcast imagery is invalid; **prefer relabeling original footage** as the clean route.

**(d) Sources.** github yastrebksv/TennisCourtDetector (+court_reference.py); arXiv 2404.06977, 2404.08401 (PnLCalib), 2207.11709 (TVCalib), 2511.04126; Farin CVIU 2008; ITF Rules; internal `court_templates.py`, `court_calibration_metric15.py`, `court_keypoint_net.py`, `court_detector_v2_schema.py`.

---

## Multi-surface (clay / grass / hard)

**(a) Carries over.** The drag+Magnus flight solver is surface-independent — only the **bounce boundary condition** becomes surface-parameterized. The trajectory-kink + audio-onset bounce *detector* (WHEN) carries; only the post-bounce state update (velocity/spin transform) needs the new model. Trust-banded advisory discipline is exactly right for clay ball-mark "review" vs Hawk-Eye/Foxtenn officiating. TrackNet variants are already reported trained across hard/clay/grass, so the ball architecture is proven multi-surface. Court geometry is regulation-fixed across surfaces, so only the visual keypoint-finder's appearance generalization changes.

**Central counterintuitive fact:** clay is "slow" because of high horizontal friction (μ≈**0.8** — *corrected from 0.8–0.9*) that scrubs pace, yet it bounces **higher** because its vertical COR is largest (≈0.83 clay vs ≈0.80 hard vs ≈0.73 grass). Grass is "fast" because low friction (μ≈**0.6** — *corrected from 0.5–0.6*) lets the ball skid low.

**(b) Must change / build.**
- **Surface-indexed bounce model** {e_v, μ} + spin-dependent grip/slide regime (Cross model: slide on low-friction grass/shallow angles; grip/kick on high-friction clay with topspin). *(medium / research-open)*
- **Surface classifier** (clay red-orange / hard blue-green-gray / grass green) as a cheap early CNN selecting physics + coaching params. *(small)*
- **Surface-robust ball detection:** optic-yellow contrast collapses on green grass (the historical reason balls went yellow in 1972), felt stains reddish on clay, per-surface shadows. *(medium)*
- **Clay ball-mark detection + advisory review overlay** (position in/out + skid-length as a spin/angle proxy) — the unique shippable tennis feature with no pickleball analog. *(research-open)*
- **Surface-indexed coaching ranges** (bounce-height, effective pace, sliding footwork, low-slice value on grass, kick-serve payoff on clay). *(medium)*
- **Adopt ITF Court Pace Rating** as the canonical surface knob: CPR = 100(1−μ) + 150(0.81 − e_T), 5 categories (Slow ≤29 … Fast ≥45), mapped to {e_v, μ} presets. *Correction:* distinguish **CPR (lab pendulum/ITF classification)** from Hawk-Eye's **CPI (in-match)** — the formula/bins here are the **CPR** standard. *(small)*

**(c) SOTA.**
- **ADOPT — ITF Court Pace Rating methodology** (public ITF standard).
- **ADOPT — Cross tennis-ball bounce physics** (academic; grip/slide transitions, COR/friction values — the standard reference).
- **ADOPT — yastrebksv/TennisCourtDetector** (open, verify license) as a cross-surface court-keypoint reference.
- **WATCH — RTMDet-Light multi-shadow detector** (Zhu et al. 2025, IET; unverified technique, no confirmed weights).
- **SKIP — Foxtenn "Real Bounce" / Hawk-Eye ELC** (proprietary; benchmark/positioning only — Foxtenn first clay ELC, Madrid 2021).

**Corrections applied.** **3DTennisDS is a Vicon 39-marker mocap dataset (4 strokes, 10 players)** — NOT a multi-surface video/ball set; largely irrelevant to surface work.

**(d) Sources.** ITF classified-surfaces + court-pace-classification; Cross (physics.usyd.edu.au/~cross/tennis.html; PUBLICATIONS/52 SpeedAndBounce); Springer athlete-surface interaction (clay 0.8/hard 0.7/grass 0.6); acsess csc2.20277 (turfgrass); foxtenn clay-ELC; ietresearch ipr2.70054.

---

## Global fusion

**(a) Carries over.** The two-step Phase-F ship is sound and carries over largely intact: **PF-1** bounded confidence-gated post-hoc nudges (foot_pin `apply_foot_pin_to_payload` idiom, always-emit audit JSON, cap-exceeded-SKIP), then **PF-2** whole-clip coordinate-descent joint optimizer (block-sparse scipy `least_squares` trf+huber, existing confidence fields as residual weights, JOSH contact-coupling as reference). The residual **families** are the right set for tennis: ball↔racket impact, ball↔ground bounce, foot↔ground, hand↔grip, net-height, non-penetration, scale anchors. The RK4+Magnus core, confidence-as-weights ruling (zero new modeling), whole-clip (no chunking) optimization, fail-closed occlusion handling, the F0 read-only consistency meter, the F4 verifier (`assert_cross_system_consistency`), and the ARKit-sidecar camera-lock path all reuse. JOSH remains the reference pattern; VERIFIED=0 discipline unchanged.

**What breaks is the SPEED/PHYSICS regime, not the architecture.**

**(b) Must change / build.**
- **≥120 fps (prefer 240) as a Phase-F precondition:** a 200 km/h serve = 55.6 m/s moves ~1.85 m/frame at 30 fps; below 120 fps arc association and contact localization are ill-posed. Multiplies the PF-2 variable/residual vector 4–8× — **compute, not modeling, becomes the binding constraint.** *(small)*
- **Continuous sub-frame `t_contact` per impact:** ball-racket dwell is ~4–5 ms, *shorter* than a 120 fps frame (8.3 ms) and ~1 frame at 240 fps — evaluate the impact residual at temporally-interpolated ball/racket poses. **Promote audio onset (sub-ms) to the primary contact-timing cue.** *(medium)*
- **Spinning-ball ODE:** Cd(Re) drag-crisis (Re 80k–300k) + spin axis/magnitude/decay as free variables; kill parabola-refit for tennis. *(medium)*
- **Surface-conditioned, spin-coupled bounce residual** + **clay ball-mark bounce anchor**. *(large / medium)*
- **Elliptical strung-face impact** with string-bed give (~10–30 mm), re-derived cap; **hand↔racket grip on a 27" lever** with one/two-hand anchors; **per-foot airborne/contact classifier** gating the foot residual (jump serves violate feet-planted). *(medium each)*
- **Broadcast-camera path:** per-frame 6-DoF from court-line PnP, degraded trust band, range-aware reprojection weighting (far players 13–17 m). *(large)*
- **Compute containment (research-open):** adaptive temporal resolution (full fps only in contact/bounce windows), B-spline/low-rank temporal basis for camera + body root, JOSH3R feed-forward init, run on H100. **JOSH runs at 0.8 FPS on a 4090**, so a 2400-frame 240 fps clip is naively out of the offline budget.

**(c) SOTA.**
- **ADOPT (pattern) — JOSH / JOSH3R** (no LICENSE file yet — port the idea only). ICLR 2026. *Corrections:* **W-MPJPE RICH is 132.5, not 184.3** (EMDB 174.7 correct); **drop the SHOW 262.3 / JOSH3R 334.9 figures** (don't match tables — JOSH3R is 661.7 on EMDB). Optimizer is **two-stage Adam, lr 0.07 / 0.014**, 500/200 iters — not "lr 0.01."
- **ADOPT — TT4D** (CC BY 4.0) — the only published racket-sport 4D fusion, 140+ h, spin annotations. **ADOPT — Cross bounce + Mehta/Steyn aero** (Steyn Cl already in repo).
- **ADOPT — PromptHMR** (research; best CalTennis translation 0.942 m). **WATCH — WHAM** (foot-contact head reusable as the residual gate), **"Where Is The Ball"** (CVPRW 2025 — *not just a preprint*; abstract does **not** confirm drag/Magnus/spin modeling → unverified for spin physics), **BlurBall**.
- **Corrections:** **WorldPose is a 2022 FIFA World Cup SOCCER dataset (arXiv 2501.02771), NOT tennis** — cite only as a transferable global-pose/broadcast-camera proxy. **CalTennis is CC-BY-NC-4.0.**

**(d) Sources.** arXiv 2501.02158 (JOSH), 2606.20542 (CalTennis), 2605.01234 (TT4D), 2504.10035 (TT3D), 2506.05763, 2509.18387, 2501.00163 (Steyn), 2501.02771 (WorldPose); Cross bounce PUBLICATIONS/52; github genforce/JOSH.

---

## Coaching

**(a) Carries over.** The coaching pillar transfers better than any other **because its method was borrowed FROM tennis** — the format-locked, no-fabrication LLM-coach design is derived from "Talking Tennis." The entire 3-stage architecture (deterministic feature extractor → non-LLM rule/reference-range comparator with typed findings + evidence pointers → format-locked LLM that never sees a raw number), the `reference_ranges` JSON schema with per-range provenance/confidence tiers, the validator (`validate_reference_ranges.py`), the `shot_rules.py` declarative rule-table evaluator, the fabrication firewall (citation cross-check + 0/300 audit `audit_coaching_fabrication.py`), and abstention discipline all carry over at the contract level. Only the ROWS and taxonomy change.

**The big tennis reversal:** pickleball found *no* usable external reference library. Tennis has the **Match Charting Project (MCP)** — **17,808 matches / 2,772,312 points / 10,503,055 shots** (*corrected: "as of early 2026," precise counts*) with per-shot type/direction/depth/error coding — plus ATP/WTA aggregates. So P6-3 shifts from "no data, build from scratch" to "abundant *pro* data, but pro-only + NonCommercial + human-charted."

**(b) Must change / build.**
- **Re-author the stat layer** to tennis canon (first/second-serve %, 1st/2nd-serve points won, ace/DF rate, winners & unforced errors + W:UE, break-point conversion/saved, net-point win %, return-points won, rally-length distribution); delete third-shot-drop/dink/kitchen/two-bounce. *(medium)*
- **Change the skill_band axis** from pickleball's 3.0/3.5/4.0/4.5+ to a tennis system — recommend **UTR (modified-Elo 1.00–16.50)** primary + NTRP coarse map + WTN note; add mph/degrees/percent/count to the unit enum. *(small / trivial)*
- **Seed P6-3 from MCP + ATP/WTA** but **facet-flag every row `pro_level`** and pair with an owner longitudinal self-baseline; **self-relative framing becomes the default voice** (comparing a rec 3.5 to Djokovic distributions is demoralizing and invalid). *(medium)*
- **Adopt MCP's stroke/error/direction notation** (n/w/d/x, !, @/#/*, serve 4/5/6) as the taxonomy for interoperability. *(small)*
- **Tennis technique metric families** (racket-head speed, contact height, swing path, unit turn / shoulder-hip separation, split-step timing ~80 ms before opponent contact, kinetic-chain sequencing) — several require the BODY-mesh + racket-6DOF pillars, so **coaching is only as trustworthy as upstream 3D.** *(large)*
- **Automated forced-vs-unforced-error rule with an explicit trust band** — UE is human-subjective (even pro charters disagree); an automated UE that silently disagrees with the player is a trust-killer. Validate agreement vs MCP human UE labels; if low, demote to "error under low pressure." *(research-open)*
- **Surface facet** on ranges + features. *(medium/later)*

**(c) SOTA.**
- **ADOPT — Talking Tennis** (arXiv 2510.03921; no code/weights; THETIS-based). *Corrections:* the **100% no-fabrication over 317 outputs IS verified in the PDF** (§4.3), but over isolated THETIS Kinect strokes with only **3 stroke types coach-evaluated**, not match narration. **Do NOT cite "coaches 8.4–8.9/10"** — that conflates a cherry-picked subset with skill scores; the actual feedback-quality ratings span **7.3–9.3/10 across 3 coaches/3 stroke types**, and the coaches' own *skill* scores were 4.0/6.9/4.7. **THETIS 12 classes = 4 backhands, 4 forehands, 3 serves, 1 smash** (the paper's own "3 backhands" prose sums to 11 — use the corrected count).
- **ADOPT — MCP analytics method** (**CC BY-NC-SA 4.0 — NonCommercial**, author-policed). **ADOPT — CoachMe/MotionExpert** (code public — verify license; +31.6–58.3% over GPT-4o on its coaching metric). **ADOPT — Kovacs 8-stage + Elliott ITF biomech** (free facts). **ADOPT — Brain Game Tennis serve+1** (proprietary content, quotable facts: 3rd-shot forehand wins 57.5% vs backhand 50.9%; 0–4-shot points ~59–70% of points).
- **WATCH — SportsGPT/KISMAM+SportsRAG** (arXiv 2512.14121, code unconfirmed), **T3Set** (KDD'25, table tennis). **Adversarial finding: off-the-shelf MLLMs hallucinate on tennis video (arXiv 2507.02904)** — reinforces the deterministic-first firewall; do not lean on MLLMs to backfill missing 3D.

**License landmine.** MCP + ATP/WTA are **CC BY-NC-SA (NonCommercial + ShareAlike)** and THETIS is research-only — none can ship inside a paid product's range library. Safe path: derive **non-copyrightable aggregate statistics** into a clean-room library with citations for internal/advisory use, never redistribute rows; get counsel sign-off. **Rec-level level-keyed distributions do not exist publicly** — same gap pickleball had; the moat is owner captures + credentialed-coach sign-off (`signed_off_by` stays null until then).

**(d) Sources.** arXiv 2510.03921 (+PDF §4.3), 2509.11698, 2512.14121, 2507.02904, 2605.12799; github JeffSackmann/tennis_MatchChartingProject, tennis_atp; THETIS-dataset; tennisabstract charting/glossary; UTR/NTRP/WTN specs; ITF Coaching & Sport Science Review.

---

## Product / competitive

**(a) Carries over.** The single-phone value prop maps directly (anti-"smart court" positioning). The advisory-never-officiating discipline + pre-registered ledger become *more* valuable as honest differentiation and match FTC substantiation doctrine. The 3-stage grounded-LLM coaching stack is the layer **no competitor ships** (SwingVision/Wingfield/Zenniz all stop at stats/charts). The full-3D-world architecture (player mesh + 6-DOF racket + ball + JOSH fusion) and **camera-motion tolerance (ARKit pose)** — which unlocks the broadcast corpus — are the categorical moats no fixed-camera rival can touch. The freemium L0/L1/L2/L3 tier ladder maps onto SwingVision's free-tier→$179.99 funnel. The iOS capture stack + Stripe/Atlas/S3 billing infra support a subscription business day one.

**(b) Must change / build (strategy).**
- **Reposition AWAY from line-calling.** SwingVision owns cheap on-device ELC (500M-shot moat, 97% within 10 cm, USTA-sanctioned); Hawk-Eye owns $40–100K/court officiating. **Cede ELC, make 3D-world + grounded coaching the hero, keep line calls explicitly advisory.** *(small)*
- **Re-underwrite the deck:** SwingVision is Apple-featured, ~$10M-raised (Series A Oct-2023, **Sony an investor**), on-device real-time, 200k+ players — a far stronger incumbent than pb.vision. It **already added pickleball** and can attack downmarket. *(medium)*
- **COGS asymmetry is the real constraint:** SwingVision's on-device inference is ~zero marginal cost; our only gate-passing tier (L3 SAM-3D-Body) is cloud GPU. **Ship on-device L0/L1 advisory for parity; reserve cloud L3 as the premium hook**, gated to user-selected rallies/serves. *(medium)*
- **Anchor pricing to $14.99/mo, ~$149–180/yr** (SwingVision parity, not pickleball's lower band); stand up **B2B seat-licensing** (academies, college/HS teams, club API); **go international early** (tennis TAM is global — US 22.5%, China 19.8%, India 9.2% player-share *— verify per-country splits against the full ITF report; US 22.5% is confirmed*). *(large)*
- **Prove the mesh is a felt benefit** (exportable 3D or a coaching insight a 2D overlay provably cannot produce — kinetic-chain/hip-shoulder separation, contact-point-in-3D, racket-face angle at impact). *(large)*

**Live-tier engineering deltas.** Decouple ball from `LiveDetectionCadenceScheduler` — **run ball every frame** through a 3-frame temporal tracker (150 mph = 67 m/s → ~1.1 m/frame at 60 fps; every-4th cadence is fatal); keep every-4th for players. **240 fps is capture-only** (a 240 fps live loop needs <4.17 ms/frame; the measured partial loop is already 4.59 ms → live per-frame caps near 60 fps). Fix the **960 px YOLO ANE-compile failure** or route ROI-640 (the 78 ft court puts the far ball at fewer pixels). Run a **≥90-min outdoor thermal soak** (the largest *unmeasured* risk — matches are 2–3 h). *Correction:* do **not** claim SwingVision "solved" 2 h sustained thermal — no primary source substantiates it; SwingVision even warns accuracy degrades facing the sun. The 217 fps "headroom" excludes the real temporal ball tracker, pose, tracking, render, and thermal — treat as a mirage.

**Scale/cost deltas.** A best-of-3 match is ~90 min (162k frames @30 fps); best-of-5 up to ~5 h (~1.08M @60 fps) — 7.5–50× pickleball. But tennis is only ~10–25% ball-in-play, so rally-gating is *more* valuable. Un-batched BODY floor (~300–400s/10s clip, BODY 96–98% of E2E) makes a rally-gated match ~$11–29 in H100-spot — upside-down vs SwingVision-free and pb.vision's $8/hr-HD ($12/hr-4K) anchor. **P5-7 BODY batching (warm-pool + multi-clip + Fast-SAM-3D-Body ~10.9×) becomes mandatory** (target ~$1.4–3.6/match). A 4K60 match is ~14–17 GB → **client-side rally-gated upload is a hard requirement.**

**(c) SOTA / market.**
- **WATCH — SwingVision** (proprietary; $179.99/yr, free 8h/mo). ~20,000+ paying subs; **2024 revenue ~$2.75M** (*corrected from "$2.5–4M ARR"; upper end is a forward run-rate estimate at best*); 100 college teams; 500+ USTA matches. The 500M shots / 200k players are **user phone captures** (its in-domain flywheel); **pursuing** ITF cert, not certified.
- **WATCH — SportAI** (B2B API; $3M round Nov-2025; MATCHi 17k courts) — racing to lock the club channel. **WATCH — Baseline Vision** (portable twin-camera 3D).
- **SKIP — Hawk-Eye/ELC** ($40–100K/court, capital-gated), **PlaySight/Wingfield/Zenniz/In-Out** (fixed-camera, assume calibrated install). **SKIP (narrative asset) — dead sensor rackets** (Babolat Play/Sony/Zepp backends shut 2020–2021, 400k+ users stranded — camera-CV won the analytics war).
- **ADOPT — RacketVision/WASB multi-sport training recipe** (+19.2% tennis mAP from multi-sport co-training). *Correction:* attribute the mAP 81.9 / P 0.945 / MDE 1.96 px to the **multi-sport MS-TrackNetV3** row (Table 3), not WASB; on the single-sport tennis benchmark TrackNetV3 beats WASB on mAP (68.7 vs 66.0) and pixel error (1.66 vs 3.62 px).

**TAM honesty.** 106M global players (ITF 2024, +25.6% since 2019) vs US pickleball ~19.8M / US tennis ~25.7M — *correction:* the "2–5×" multiplier compares **global tennis to US pickleball**; US-to-US is ~1.3×. Academy economy ~$8.4B (Dataintelo vendor estimate). Most participants are casual beginners (70% beginner/intermediate) with low WTP — bottom-up serviceable TAM is far smaller than the 106M headline.

**(d) Sources.** ITF Global Tennis Report 2024; SFIA 2024; swing.vision (+ Sportico Series-A, Sony Innovation Fund); sportai.com; hawkeyeinnovations.com; auratidecollective (dead sensors); arXiv 2511.17045; runs/ios_device_gate LATENCY_TABLE; TennisExpert arXiv 2603.13397; pb.vision API guide.

---

## Data / datasets / broadcast engine

**(a) Carries over.** The WASB + blurball + TrackNetV3 zoo (TrackNet *born* on tennis broadcast), `court_unet_v2` + homography + net-plane, the drag+Magnus arc solver + synthetic-lift, Fast SAM-3D-Body + YOLO26m + BoT-SORT/OSNet, the 3-stage grounded-LLM scaffold, the held-out ledger + SST teacher-student loop, and the paddle-6DOF detection scaffold all transfer. Broadcast tennis has a **fixed elevated single-camera geometry over a known court**, so auto-homography gives near-free frame-level 3D-court anchoring pickleball never had — a genuine flywheel for **physics/geometry priors**.

**Net estimate:** tennis removes roughly **60–70% of the pickleball labeling burden** (court + 2D ball + coarse stroke priors) but the two highest-value moats — **own-camera in-domain temporal data** and **any 3D ground truth** — still require owner capture + the same SST loop.

**(b) Must change / build.**
- **Build a broadcast-harvest data-engine lane** (YouTube ingest → rally seg → auto-homography → WASB/TrackNet ball → trajectory-kink bounce → PaddleOCR scoreboard + serve-speed). Target **physics/geometry/event outputs, NOT phone-domain detector weights.** *(large)*
- **Design for TWO flywheels:** broadcast for physics/motion/eval/reference priors that transfer *through 3D geometry*; **owner/user phone-capture SST for the detector domain that does not.** The appearance domain gap (high tight TV framing, tiny blurred ball vs fence-mounted iPhone) is the exact Roboflow-transfer death measured 4× in pickleball. **Do not let broadcast abundance de-fund phone capture.** *(medium)*
- **Real-3D-arc seed:** recover metrically-scaled pro trajectories from broadcast via auto-homography-anchored single-view physics fit — replacing the synthetic-lift-only seed (highest-transfer use because it flows through geometry). *(large)*
- **Data-driven coaching reference library** from broadcast + MCP/ATP/WTA. *(medium)*
- **Legal/licensing pass:** quarantine unclear/NC data from the shippable model path (see below). *(small)*
- **Pre-registered domain-gap probe FIRST:** train ball/player detectors on broadcast, eval on phone-domain held-out, quantify the drop — decides how much broadcast can do. *(small)*

**(c) SOTA / datasets (corrections applied).**
- **ADOPT — TrackNet (v1) tennis ball data** (research; **no clear open license**). *Correction:* the **36,962-frame set is the ORIGINAL TrackNet (Huang 2019** — 20,844 frames from the 2017 Universiade men's final + 16,118 court-setting frames), **NOT "TrackNetV2's dataset"** (V2's own release is **badminton**, 55,563 frames). Lab is now **NYCU** (ex-NCTU); no LICENSE file → treat as restricted.
- **ADOPT — TennisCourtDetector dataset** (8,841 imgs, all surfaces; license unspecified → research/seed only, not shippable — and the Gholamreza HF "MIT" re-upload does not confer commercial rights to the underlying broadcast frames).
- **ADOPT (immediate) — TennisVL/TennisExpert** (github LZYAndy/TennisExpert; **202 broadcast matches / 471.9 h / 40,523 rally clips**, 94 players, 162,503 shots). *Correction:* "YouTube / singles / scoreboard / 2019–2025" qualifiers are **not confirmed by the abstract** — soften to "broadcast matches."
- **ADOPT (coaching) — MCP + tennis_atp/wta** (**all CC BY-NC-SA — NonCommercial**, author-policed). **CC BY sets (Mendeley 2024 pose, 2,000 imgs, 17 kpt)** and owner-captured data are the clean production route.
- **WATCH — Vid2Player3D** (**"NVIDIA License" — NonCommercial**; trained models withheld for footage copyright). SIGGRAPH 2023 Best Paper HM. Code released, **not a redistributable corpus** — method blueprint only; reimplement with commercial-clean components (ViTPose Apache-2.0).
- **WATCH — AthletePose3D** (CVSports@CVPR-2025, optical mocap, ~1.3M frames, **non-commercial, NOT tennis-specific**). **SKIP — arXiv 2511.04126** (marketing-grade, no held-out ledger).
- **THETIS** (8,374 clips, 55 subjects, 12 strokes; **Kinect depth-skeleton quality, research-only** — not high-quality 3D pose GT).

**Adversarial evidence the gap is real:** SwingVision's ~97%-within-10 cm edge is built on ~**500M shots from ~200k players' PHONE captures**, not broadcast — the market leader chose phone data for a reason.

**(d) Sources.** nol.cs.nctu (TrackNet); arXiv 1907.03698, 2311.05237, 2603.13397 (TennisVL), 2511.17045; github JeffSackmann/*, THETIS-dataset, nv-tlabs/vid2player3d (+LICENSE.txt); Mendeley nv3rpsxhhk; ITF Rules.

---

## Eval / ground-truth

**(a) Carries over.** The entire eval discipline transfers verbatim and becomes MORE valuable (more real GT = more surfaces to accidentally leak): the pre-registered held-out ledger (`heldout_eval_ledger.md`), `eval_guard.py`'s `assert_not_training_on_eval_clip` leak protection + `EvalClipLeakError` registry, once-only scoring, `append_lock.py` atomic append, the PREREGISTRATION go/no-go pattern, and **VERIFIED = only-a-passed-gate-on-real-labels**. The ball-2D metric+gate machinery (micro F1@20px, recall@20px, hidden-FP-rate, teleport-count), PCK@Npx for court keypoints, the 5-keypoint racket → IPPE architecture (≈ RacketVision task), and the zero-fabrication 3-stage coaching scaffold all reuse.

**Tennis inverts the pickleball GT starvation.** Pickleball sits at VERIFIED=0 with only ~770 visible + 381 hidden owner-drawn ball boxes on two clips and *zero* real 3D labels. Tennis has physically-calibrated GT for nearly every pillar — **most de-risking exactly the two pillars pickleball is worst at (3D ball flight, court calibration)**, which can for the first time be scored against real metric truth.

**(b) Must change / build.**
- **Stand up an our-camera strict held-out set** (not broadcast) stratified by surface (clay/grass/hard) and singles/doubles, with CVAT ball+court+player labels — the asset that lets us claim an honest number for consumer tiers. *(large)*
- **New gates pickleball could never define:** serve-speed error vs broadcast radar (~±1 km/h device accuracy); 3D bounce-location error vs Hawk-Eye / clay ball-mark; real world-MPJPE vs marker mocap. *(now/soon)*
- **Domain-gap protocol:** label any metric earned on broadcast/Hawk-Eye/sensor GT **REFERENCE-VERIFIED**, distinct from **OUR-CAMERA-VERIFIED**; only the latter backs a product claim. Wire tennis clip IDs into `eval_guard`'s strict-holdout registry on day one so partnership/broadcast GT cannot leak. *(now)*
- **First-ever real BODY gate:** world-MPJPE / joint-angle correlation vs marker serve mocap. *(soon)*

**(c) SOTA / GT sources (corrections applied).**
- **ADOPT — Hawk-Eye/ELC as reference 3D ball+bounce GT** (proprietary; **partnership-gated**, e.g. Tennis Australia Game Insight Group — **not a public download**). Mean error ~3.6 mm; ball at 340 fps. *Correction:* GIG **player position is (x,y) at ~20 fps (50 ms)** — the millisecond granularity applies to the **ball**, not player feet.
- **ADOPT — broadcast serve-speed radar** (±1 km/h) as scalar ball-speed GT. **ADOPT — marker-based serve mocap** (biomech gold standard; joint-angle CIs 5.1–30.8°). **ADOPT — RacketVision** (MIT; racket PCK@0.2 89.6%). *Correction:* the tennis ball row is **P 0.962 / R 0.797 / MDE 1.66 px / mAP50 68.7** (single-sport, bg-modeling, 4 frames) — verify which variant the P0.945/R0.880/MDE1.96 figures cite.
- **ADOPT — WASB / TrackNetV3; PnLCalib-style point+line court registration** (the killed heatmap `court_unet_v2` — owner-clip PCK@5 ~0.017 — is what this replaces, now anchored to exact ITF geometry).
- **WATCH — sensor rackets** (racket swing-speed **ICC=0.983** vs 500 Hz VICON, but impact-location **weak: κ 0.21–0.41** — NOT a gate; **hardware effectively dead 2020–2021**). **WATCH — SkeleTRACK** (proprietary, no published MPJPE vs mocap — marketing-grade). **SKIP — SwingVision** (competitor trust bar).
- **ADOPT — SportsPose** (markerless-vs-marker 34.5 mm) / **AthletePose3D** (fine-tuned ~65–99 mm, **not tennis, non-commercial**) as pose priors. **License traps:** MCP + tennis_atp are **CC BY-NC-SA**; RacketVision MIT-stated (verify no broadcast-rights caveat).

**(d) Sources.** ITF Rules; en.wikipedia Hawk-Eye; tandfonline 24748668.2023.2291238; GIG academic-industry partnership; tennis.com radar-gun; pubmed 30540215 (sensor validity); S0021929025006311 (serve mocap CIs); arXiv 2511.17045, 2503.07499; SportsPose; internal `eval_guard.py`, `heldout_eval_ledger.md`, `ball_detector_gate.py`.

---

## Legal-IP

**(a) Carries over.** The **advisory-not-officiating** rule is our single biggest de-risking asset and maps directly onto FTC substantiation doctrine + the officiating-claim boundary. The pre-registered ledger + "VERIFIED = passed gate on real labels" maps onto FTC's competent-reliable-evidence standard (cf. the Workado 2025 order over an unsubstantiated 98% claim). The **monocular single-iPhone architecture is a deliberate design-around** of Hawk-Eye/Sony's multi-camera triangulation claims. The internal-use-only posture buffers both patent risk (no offer-to-sell) and training-data risk (no redistribution). Per-component license hygiene (MHR Apache-2.0 vs SAM-3D-Body custom) extends unchanged.

**Tennis is a materially harsher IP climate — patented AND actively litigated.**

**(b) Must change / build (all launch-blocking gates).**
- **Commission an FTO opinion against SwingVision US11893808B2** ("Learning-based 3D property extraction," priority 2020-11-30, granted 2024-02-06, Mangolytics→SwingVision) — it claims a **single neural net on a mobile camera extracting a ball's 3D property/landing from monocular 2D video**, i.e. almost exactly our Ball-3D + in/out pillar, owned by the Sony-backed market leader. *(large)*
- **Clear Infinity Cube's patent** — *correction: verified as **US 10,467,478** ("Method for Mobile Feedback Generation Using Video Processing and Object Tracking")* — asserted in **Infinity Cube Ltd. v. Mangolytics, S.D. Cal. 3:22-cv-00547** (filed Apr 2022). *Corrections:* Infinity Cube's ITF-approved product **eyes3 requires ≥2 iPhones** (not "single-camera"), and holds **ITF PAT approval (2019)** — distinct from the current Gold/Silver/Bronze ELC scheme. Case outcome remains genuinely unverified; the filing alone proves competitors will sue over app-based line-calling. *(medium)*
- **Purge "officiating / official / line calls / ITF-certified / Hawk-Eye-accurate / replaces the umpire"** from all tennis copy → "advisory line-call assistant / practice tool" + visible trust band + not-for-sanctioned-officiating disclaimer. The ITF Gold/Silver classification + USTA Regulation III.C **legally gate "officiating"**; **SwingVision is NOT ITF-classified**, so no monocular consumer app may brand itself certified. *(small)*
- **Purge broadcast/YouTube footage from commercial training** → owner-captured + CVAT + SST, and/or explicitly licensed corpora. Broadcast carries broadcaster copyright + league/EU database rights + player publicity/likeness, and ATP/Wimbledon official-data deals (TDI–Sportradar) **expressly ban "collating data from TV pictures"**; US Copyright Office (2025) says wholesale training copying "ordinarily weighs against fair use." *(large)*
- **Substantiation packet for every accuracy figure** (sample, surface, camera setup, near-line band, ledger row) + **trademark/publicity clearance** for player names / tournament marks / comparative "as accurate as Hawk-Eye" claims. *(medium/small)*

**Officiating-context corrections.** The correct 2025 ELC milestone is the **ATP Tour mandating ELC (incl. clay)**, **not** Roland Garros (which kept human line judges). Hawk-Eye's documented figure is the **3.6 mm advertised average** (drop the unverified 2.2 mm). Sony **is** a SwingVision investor (seed Apr 2022 + Series A), which is exactly why an FTO is mandatory.

**(c) SOTA / IP landscape.**
- **WATCH — SwingVision US11893808B2** (proprietary; granted 2024). Closest patent to our architecture; blocking-patent risk. Marketing metrics (97%@10cm, 500M shots) are **company claims, not independently verified.**
- **WATCH — Infinity Cube US 10,467,478 / eyes3** (proprietary; ITF PAT-approved 2019; ≥2-iPhone). Proves willingness to litigate.
- **WATCH — Hawk-Eye/Sony estate** (proprietary; foundational GB 9929193.2 / WO 01/41884 **lapsed**; US 9,646,382 is placard-detection). Core claims lean **multi-camera triangulation** — our monocular design likely sits outside the strongest claims, but FTO still required (layered patents + trade secret + know-how).
- **WATCH — ITF ELC Gold/Silver scheme** (Gold: Hawk-Eye, Foxtenn, IMG Arena, Bolt6 Sentinel; Silver: PlayReplay, Zenniz — **no app-only monocular system classified**). **SKIP — Foxtenn** (ITF Gold; reframed as pioneering clay system, not sole tour-approved).

**Risk posture.** Internal-use-only lowers but does not eliminate risk — US patent law reaches "make/use" (§271(a)); the experimental-use defense is narrow. Every commercial-tennis gate (FTO on '808/'478, zero official/ITF claims, substantiated accuracy, clean-rights training data) must be **green before launch**.

**(d) Sources.** patents.google US11893808B2, **US10467478**; Bloomberg Law / eyes3.com (Infinity Cube v. Mangolytics 3:22-cv-00547); marks-clerk Hawk-Eye IP; itftennis classified-ELC + ELC-evaluation-paper; usta ELC / Regulation III.C; sportradar ATP addendum; ftc.gov Workado 2025; copyrightalliance / wiley Copyright-Office 2025 fair-use; arXiv 2107.09255.

# PART II-B — PICKLEBALL → TENNIS DELTA MATRIX

The single most useful engineering artifact in this document: per pillar, what **reuses** the existing
pickleball code/architecture vs. what must be **changed or newly built** for tennis, with effort and
risk. "Effort/Risk" reflect the dominant change items per pillar; "research-open" items (axial limb
rotation, spin RPM, strung-face blur, PF-2 compute) are the ones most likely to stay UNVERIFIED at
launch and must ship advisory-banded. The lowest-effort/highest-certainty wins are the **scoring FSM**
(pure deterministic software) and the **court template** (already correct in-repo); the highest-value
new build is the **court-keypoint heatmap CNN** — the exact architecture pickleball *killed* for data
scarcity (PCK@5 0.017–0.056), now viable because tennis has 8,841+ public labels. Three pillars are
net-new (no pickleball analog): serve biomechanics, the match-scoring FSM, and multi-surface/ball-mark
review.

| Component / Pillar | Reuses from pickleball platform | Must change / newly build for tennis | Effort | Risk |
|---|---|---|---|---|
| **Ball — 2D detection** | WASB-SBDT (MIT, HRNet, 3-frame heatmap) as default; blurball train fork + `build_ball_blur_sidecar`; TrackNetV3 + union candidate pool; `benchmark_ball_trackers.py` harness; SST + CVAT loop; top-K sidecar + physics-fill; trust bands | Swap pickleball fine-tunes for **upstream tennis WASB/TrackNet checkpoints** (tennis is *in-domain* for TrackNet — born on it); re-derive teleport gate (160 px/frame rejects every serve); tiled/hi-res inference for far-baseline ball (78 ft court); retire TrackNetV4 (undeserializable weights, F1 0.9581 < V2 0.9677); modest phone-view fine-tune (100s–1000s frames) | **Medium** | **Low–Med** — tennis is data-rich (inverts the pickleball 0.6969 zero-shot wall); risk is broadcast→phone domain gap, but ~10× less data than pickleball needed |
| **Ball — 3D flight / spin** | RK4 drag+gravity+Magnus ODE core (`_rk4_step_with_magnus`); event-anchored per-segment BVP arc solver; synthetic-lift (TT4D/UpliftingTT) pattern; size-depth residual; `spin_rpm` schema slot exists | Re-seed felt-ball physics (Ø 65.41–68.58 mm, ~58 g; **Cd ~0.507±0.024 banded, contested vs. Mehta 0.6–0.7**); retune Magnus to Cl≈0.55·S (pickleball 0.195·S); **raise 35 m/s speed ceiling** — serves are 54–67 m/s and silently rejected today; promote `spin_rpm` from `None` stub to solved variable; alt-aware ρ. Spin is net-new (currently zero spin estimate) | **Large** | **High** — Cd genuinely contested/wear-dependent; zero real 3D labels persist; per-shot RPM needs owner radar GT (TrackMan/Pocket Radar); full 3D axis not monocular-recoverable |
| **Bounce / contact / line-call** | `court_templates.py` already ships correct tennis template; `classify_ball_line_calls`, σ_bounce elevation-parallax model, `too_close_to_call` band, event_fusion (audio+kink+wrist); `_has_excess_bounce` = tennis double-bounce | Swap `PICKLEBALL_DROP_TEST_HEIGHT_M`→ITF rebound (254→135–147 cm); **mandate ≥120–240 fps** (150 mph serve = 1.8–2.2 m/frame @30 fps → σ_depth meters); add service-box/singles-vs-doubles line context, service-fault call; surface-conditioned restitution/friction; **clay ball-mark detection** (new pillar); retune audio for ~5 ms strung "thwock" | **Large** | **High** — speed×framerate is the killer; clay unsolved single-cam; over-claim liability since ATP mandated ELC on clay 2025 (Madrid/Rome) |
| **Body — 3D / mesh** | Fast SAM-3D-Body + MHR70 (Apache-2.0 rig); classical world-grounding (LK+MAD, foot-plane, MAD+Gaussian smoothing); YOLO26m + BoT-SORT/OSNet; MHR-latent smoothing; SOMA-X SMPL-X pivot | **Phase-gate OFF foot-lock during serve flight** (both feet airborne — highest-severity bug); per-phase adaptive smoothing (W=9 erases serve snap: shoulder IR ~2400°/s); **axial limb rotation** as first-class output (research-open, HMR's weakest DOF); racket-occlusion + two-handed-BH grip; re-anchor scale to tennis court; **first-ever GT validation** via CalTennis + AthletePose3D | **Large** | **Med–High** — first real BODY gate becomes possible (pickleball never GT-validated); axial rotation may stay UNVERIFIED without racket-6DOF fusion |
| **Serve (NEW PILLAR)** | 3-stage coaching pipeline; `contact_windows` fusion; foot-contact + court calibration; iOS 240 fps sidecar already shipped; ball zoo for toss | 8-stage phase segmenter (Kovacs-Ellenbecker); **240 fps hard precondition** (accel phase <0.01 s, unsamplable at 60 fps); serve reference library (max ER 172±12°, contact 38–47 m/s, GRF 1.68–2.12 BW); ring-fence distal velocity/racket-head speed as advisory; foot-fault detector; injury markers as non-clinical proxies only | **Large** | **High** — the headline number (racket-head speed via shoulder IR ~2400°/s) is the *least*-reliable monocular signal; SwingVision/Hawk-Eye already ship serve speed |
| **Stroke classification** | Abstain-below-confidence discipline; per-shot record schema; WiLoR hand crops; SST + CVAT; 3-stage LLM validated by *Talking Tennis* | **Stop deriving stroke from ball geometry** (`_classify_shot_type` is trajectory-only, cannot see FH/BH); new pose+trajectory pillar (port BST fusion); replace pickleball enum (dink/erne/atp) with tennis taxonomy; 1H-vs-2H BH + handedness; topspin/slice *inferred* from swing path (spin sub-frame at 30–60 fps); macro-F1 gating for class imbalance | **Large** | **Med** — THETIS/TenniSet are lab/broadcast, off-distribution + research-licensed; monocular far-player self-occlusion |
| **Racket 6-DOF** | Full `paddle_pose_fused` (MHR hand-frame H(t) + Wahba grip-transform G + ball-reflection + SLERP one-euro); 5-keypoint schema ≈ RacketVision (top/bottom/handle/left/right); IPPE planar-PnP (strung face is coplanar ellipse — valid) | Geometry: 16×8″ rectangle → **~27″ elliptical strung head** (~318×265 mm), long lever (t_g 3–4× longer → amplified tip jitter); tennis grip-prior library (Eastern/SW/W/Continental/serve-pronation); **two-handed-BH dual-wrist model**; lever-arm racket-head speed (v_head = v_wrist + ω×r); contact-point-on-string-bed; retrain detector (RacketVision seed) — lift weak face-width keypoints | **Large** | **Med–High** — physics-inversion face angle bounded ~26° (TT4D); strung face = weak silhouette; broadcast RacketVision seed won't transfer zero-shot |
| **Court / net geometry** | `court_templates.py` tennis entry already correct (78×36 ft, 27 ft singles, 21 ft service line, 42″ post/36″ center); `court_zones.py` service boxes + alleys; metric-15pt PnP (8 px/15 px gates); homography + `net_plane` API; calibration stack + CVAT loop | Replace 15-name pickleball keypoint schema → **tennis 14–16 pt** (service boxes, T-junctions, singles sidelines, no kitchen); **singles-vs-doubles runtime classifier** (new — same lines both formats); piecewise net model (center-strap 0.914 m + singles-stick double-dip); surface-aware line-contrast; **don't ship yastrebksv weights** (no license, broadcast domain) — reimplement | **Med–Large** | **Med** — most *de-risked* failing pickleball pillar (court_unet_v2 died at PCK@5 ~0.017); tennis data-rich; far-corner off-frame more common at 78 ft |
| **Multi-surface (clay/grass/hard)** | Drag+Magnus flight (surface-independent); bounce *timing* detector; SST/CVAT; court geometry (regulation-fixed across surfaces) | Surface classifier (cheap CNN, color/texture); **surface-indexed bounce** (COR clay ~0.83 / hard ~0.80 / grass ~0.73; μ clay ~0.8 / hard ~0.7 / grass ~0.6); adopt **ITF Court Pace Rating** CPR=100(1−μ)+150(0.81−eₜ); spin-dependent grip/slide regime; surface-robust ball detection (yellow-on-grass contrast collapse); clay ball-mark as free bounce GT | **Large** | **Med–High** — single non-ground camera limits clay-mark accuracy (Foxtenn needs 40 ground cams); grass data scarcest |
| **Fusion (Phase-F)** | Two-step PF-1/PF-2 ship pattern; residual families (impact/bounce/foot/grip/net/scale); confidence-as-weights; scipy TRF + block-sparse Jacobian; whole-clip opt; JOSH reference | **Sub-frame t_contact free variable** (dwell 4–5 ms < one 120 fps frame); Cd(Re) + spin as free variables; surface-conditioned spin-coupled bounce; strung-face string-bed give (~10–30 mm); dual-wrist grip on 27″ lever; **airborne-foot gate** for jump serve; broadcast PTZ camera path; compute containment (JOSH 0.8 FPS → 240 fps × 2400-frame clips out of budget) | **Large / research-open** | **High** — compute is the binding constraint, not modeling; VERIFIED=0 on real tennis labels |
| **Coaching (grounded LLM)** | *Entire* 3-stage pipeline unchanged (design *came from* tennis Talking Tennis); reference_ranges JSON schema + validator + 0/300 fabrication audit; provenance tiers; trust bands | Re-author stat layer (serve %, W:UE, break-point, ace/DF — delete third-shot-drop/dink); skill axis NTRP/**UTR**/WTN (not 3.0/4.0); seed ranges from **Match Charting Project** (17,808 matches / 10.5M shots) — but pro-only + CC-BY-NC-SA; automated UE attribution (human-subjective) trust-banded; surface facet; ×20–40 row explosion | **Med–Large** | **Med** — best data is NonCommercial + pro-only (rec distributions don't exist publicly); UE inherently contestable |
| **Rules / scoring (NEW)** | `shot_rules.py` evaluator engine; σ/margin/`too_close_to_call` pattern; `_has_excess_bounce`; court zones already built | **Greenfield match FSM** (grep finds zero rally/score/deuce/tiebreak state) — points→game→set→match, config-driven per Appendix VI (No-Ad, tiebreak, Coman); score-driven diagonal service-box selection; serve fault→2nd→double-fault sub-FSM; refuse intent-based calls (hindrance/let); singles/doubles sideline switch | **Medium** | **Low** — the FSM is exact, 100%-testable software (the one pillar VERIFIABLE without ML); risk is garbage-in from noisy event detection |
| **iOS / live tier** | LiveFrameTap; ANE benchmark harness; coming-soon kill-switch; L0/L1/L2/L3 constitution; person track (4 players); 240 fps capture sidecar; WASB temporal family | **Decouple ball from cadence scheduler** (every-4th = 4.5 m gaps at 150 mph — fatal); train in-domain tennis ball student; resolve 960 px ANE-compile failure vs. far-ball resolution; tennis court geometry + singles/doubles toggle; advisory line-call + serve speed first-class; **240 fps = capture-only** (live loop caps ~60 fps; partial loop already 4.59 ms > 4.17 ms budget) | **Med–Large** | **Med–High** — 217 fps "headroom" is a mirage (excludes real temporal tracker/pose/render/thermal); sustained outdoor thermal unmeasured |
| **Data engine** | SST teacher-student; CVAT prelabel-then-correct; TensorRT/FP16; spot-resume; per-court warm caches (H1/H4/H6); held-out ledger discipline | Two-flywheel model: **broadcast** (TennisVL: 202 matches / 471.9 h / 40,523 rally clips) for physics/geometry/reference priors **that transfer through 3D**; **owner phone-capture SST** for the detector domain that does *not*; broadcast-harvest subsystem (ingest→homography→ball→OCR); match-level rally splitter (90 min–5 h, ~150–250 rallies); licensing quarantine (TrackNet no-license, MCP NonCommercial, Vid2Player3D withheld models) | **Large** | **Med** — broadcast abundance is a *trap* if it de-funds phone capture (SwingVision's moat is ~500M *phone* shots, per vendor, not broadcast) |
| **Eval / GT** | Pre-registered held-out ledger; `eval_guard` leak protection; once-only scoring; ball-2D metric gates; PCK@Npx; VERIFIED=passed-gate rule | Our-camera strict held-out set stratified by **surface × singles/doubles**; NEW gates impossible in pickleball — serve-speed vs. radar, 3D bounce-loc vs. Hawk-Eye/clay-mark, world-MPJPE vs. marker mocap; two-tier labeling: **REFERENCE-VERIFIED** (broadcast/Hawk-Eye) vs. **OUR-CAMERA-VERIFIED** (only this backs consumer claims); Hawk-Eye GT is partnership-gated | **Large / research-open** | **Med** — tennis *inverts* pickleball's GT-starvation, but richest GT is broadcast-domain + proprietary; more GT = more leak surfaces |

**Notes on Table A.**
- **Effort/Risk** reflect the dominant `must_change` items per pillar. "Research-open" items (axial rotation, spin RPM, strung-face blur, PF-2 compute) are the ones most likely to stay `UNVERIFIED` at launch and must ship advisory-banded.
- The **lowest-effort, highest-certainty** win is the **scoring FSM** (pure deterministic software, 100%-testable) and the **court template** (already correct in-repo). The **highest-value new build** is the court-keypoint heatmap CNN — the exact architecture pickleball *killed* for data scarcity (PCK@5 0.017–0.056), now viable because tennis has 8,841+ public labels.
- Three pillars are **net-new** (no pickleball analog): **serve biomechanics**, **match scoring FSM**, **multi-surface/ball-mark review**.

# PART II-C — TENNIS COMPETITIVE & PRODUCT LANDSCAPE + GO-TO-MARKET

Tennis is a far more crowded, better-capitalized, and IP-encumbered CV market than pickleball. The
honest headline: **SwingVision is a genuinely strong, entrenched incumbent** — nothing like the weak
pb.vision — so any tennis plan that assumes "we out-execute the leader" starts from a false premise.
Our wedge is *not* line-calling; it is the single-phone full-3D world + grounded coaching +
camera-motion tolerance + multi-surface, sold explicitly as an honest **advisory** product. What
follows: the competitor landscape, our ordered differentiators, TAM + segments, pricing/positioning,
and the honest threats — including the two launch-blocking gates (FTO on SwingVision's US 11,893,808 B2,
and the NonCommercial-data quarantine).

## Tennis Competitive & Product Landscape + Go-to-Market

Tennis is a far more crowded, better-capitalized, and IP-encumbered CV market than pickleball. The honest headline: **SwingVision is a genuinely strong, entrenched incumbent** — nothing like the weak pb.vision — and any tennis plan that assumes "we out-execute the leader" is starting from a false premise. Our wedge is *not* line-calling. It is the single-phone full-3D world + grounded coaching + camera-motion tolerance + multi-surface, sold explicitly as an honest **advisory** product.

### A. Competitor Landscape

| Product | Tech | Camera setup | On-device / Cloud | Pricing | Ships 3D / mesh? | Coaching depth | Key weakness |
|---|---|---|---|---|---|---|---|
| **SwingVision** | On-device CV: ball model trained on **500M+ shots / 200k+ players** (vendor claim); real-time Apple Neural Engine at locked 1080p/60fps | **Single iPhone** behind baseline (adding a 2nd camera for 99% close calls) | **On-device** (≈zero marginal cost) | **$179.99/yr** or $14.99/mo; free tier ~8 hrs/mo HD | **No** — 2D + ball trajectory only; patents mesh-*deformation* but ships overlay | Medium: stats, shot taxonomy, highlights, line calls; not grounded natural-language biomechanics coaching | Depth/biomech ceiling: 2D-overlay-only, no true 3D world; misreads volleys as flat forehands (anecdotal) |
| **Hawk-Eye Live / ELC (Sony)** | Multi-camera triangulation + **SkeleTRACK** 29-point skeleton (tennis debut Laver Cup 2024) | **Fixed ~6–12 (up to ~18) cameras/court**, 6 for foot-faults, 340 fps | Cloud/enterprise | **~$40k–$100k per court** install (~$25k/court at scale), ~$40k maintenance | Skeleton (29-pt), **not mesh**; broadcast/officiating only | N/A — officiating & broadcast infra, not consumer coaching | ~**3.6 mm** advertised avg error requires a fixed calibrated rig; not a consumer product; capital-gated |
| **PlaySight SmartCourt** | Fixed multi-cam smart-court, livestream + analytics (acquired by Slinger 2021) | Fixed calibrated install | Cloud | Highest-tier venue SaaS (undisclosed) | No mesh | Medium (pro-coach/college tier), labor-heavy | Fixed install; slow; no single-phone consumer story |
| **Baseline Vision** | Portable "Hawk-Eye-style" 3D-trajectory device | **Twin-camera** net-post unit | Hybrid (device + app) | Hardware + app subscription | **3D ball trajectory** (fixed 2-cam), no body mesh | Low–medium | Fixed twin-camera; ball-3D only, no player mesh/coaching depth |
| **In/Out AI Line Judge** | Portable line-calling (v4.0), single→multi-device pivot | **~3 devices** (1 Net + 2 Line) | On-device | **~$499** hardware | No | None (line calls only) | The single-cam→multi-device pivot itself is evidence single-fixed-camera line calling is geometrically hard |
| **Wingfield** | Dual-perspective 4K "smart net post," gamified training (raised €4M) | Fixed two-camera net post | Cloud | **~$5,000 hardware + $100–200/mo** | No mesh | Medium (drills/gamification) | Fixed ~$5k install; club-scale, not consumer |
| **Zenniz** | End-to-end smart court: **30 audio sensors + 4 cameras**; GEN2 clay/outdoor (2025) | Fixed multi-sensor install | Cloud | Hardware + subscription | No mesh | Medium | Fixed install; ITF **Silver** ELC tier, not consumer-portable |
| **Sensor rackets (Babolat Play / Zepp / Sony STS)** | Wrist/racket IMU swing/spin/impact | None (worn sensor) | Device sync | Discontinued | No | Swing-speed/spin stats only | **CATEGORY DEAD**: Babolat Play EOL Mar-2021, Sony STS ended 30-Sep-2021, Zepp servers offline 2020 — stranded 400k+ users. Camera-CV already won. |
| **Dartfish / Tennis Analytics (legacy pro-coach)** | Manual/semi-auto video tagging on fixed installs | Fixed / manual | Cloud/desktop | Enterprise/pro-coach | No | High but manual, labor-intensive | Slow, labor-heavy, no consumer single-phone flow |

**Reading the table:** the market splits into (1) an **officiating tier** Hawk-Eye owns via a ~$40k–$100k/court, ~6–18-camera rig at ~3.6 mm; (2) a **fixed-install "smart court" tier** (PlaySight, Wingfield, Zenniz, Baseline) that all assume a *fixed calibrated camera*; (3) a **consumer single-phone tier** SwingVision dominates on-device; and (4) a **dead wearable-sensor tier**. No consumer product ships a full player **mesh + 6-DOF racket + ball** from one handheld/moving phone — that is the open lane.

### B. Our Ordered Differentiators (tennis)

1. **Full 3D world from ONE phone** — player mesh + racket 6-DOF + ball, where SwingVision ships 2D + trajectory, Baseline does ball-3D only, and Hawk-Eye SkeleTRACK is a 29-point skeleton (not mesh) and broadcast-only. *Caveat: this is only a moat if the mesh is a felt, exportable benefit — a coaching insight (kinetic-chain / hip-shoulder separation, 3D contact point, racket-face angle at impact) a 2D overlay provably cannot produce — not a demo.*
2. **Camera-motion tolerance (ARKit pose)** — the one axis categorically ahead of the *entire* field: Hawk-Eye, PlaySight, Wingfield, Zenniz, and Baseline all require a fixed calibrated camera and **structurally cannot ingest a handheld pan or a broadcast/YouTube clip**. This should be the flagship *live demo*, not a slide.
3. **Grounded, zero-fabrication coaching** — the 3-stage deterministic-features → reference-range comparator → format-locked LLM layer that NO competitor ships; all rivals stop at stats/charts. Independently validated by the "Talking Tennis" pattern.
4. **Multi-surface (clay/grass/hard)** — surface-dependent bounce physics + clay ball-mark reasoning, a wedge none of the consumer apps cover (Zenniz even had to split GEN2 for clay).
5. **Honest advisory trust bands** — deliberately ceding officiating to Hawk-Eye/SwingVision; positioning on trust > novelty in a category that oversells "Hawk-Eye-style" accuracy.

### C. TAM + Segments

- **Global base:** ITF Global Tennis Report 2024 counts **106M players across 199 nations** (up 25.6% from 84.4M since 2019). Player-share is genuinely global — **US 22.5%** is the one firmly verified national figure; other per-country splits (China, India, GB, Germany) are directionally large but not independently confirmed here.
- **US base (apples-to-apples):** SFIA 2024 puts **US tennis at ~25.7M (+8% YoY)** vs **US pickleball ~19.8M** — so US-to-US the addressable base is only **~1.3×**, not "2–5×." The 2–5× multiplier only holds *global tennis vs US pickleball* — a base mismatch to flag, not hide. The real advantages are **global reach** (pickleball was mostly-US) and **higher ARPU**.
- **Proven WTP:** SwingVision runs **20,000+ paying subscribers** at $179.99/yr, ~**$2.75M booked 2024 revenue** (up ~12% from $2.46M in 2023; some cite $2.5–4M ARR as a forward run-rate), with **100 D1–D3 college teams** and 500+ USTA matches officiated.
- **B2B academy economy:** ~**$8.4B (2024) → $15.2B by 2033, 6.7% CAGR** (Dataintelo vendor estimate) — a coaching-mediated, institution-paid culture with structured NCAA/USTA channels.
- **Serviceable reality:** 70% of the 106M are beginner/intermediate with low WTP; bottom-up serviceable TAM is far smaller than the headline. AI-in-sports market sizings ($0.97B–$8.9B for 2024) vary wildly and should be treated as directional tailwind only.

**Segments (by WTP / CAC):** (i) **Junior academies** — highest WTP, lowest CAC, institution-mediated, doubles as a data-capture channel; (ii) **College / HS teams** — SwingVision already at 100; (iii) **serious adult league / UTR players** — the D2C core at $150–180; (iv) **casual improvers** — free-tier funnel; (v) **clubs** — contested channel SportAI is racing to lock via MATCHi (1M+ users, 3k venues, **17k courts, 30 countries**).

### D. Pricing / Positioning Options

Anchor to the **tennis WTP band, not pickleball's**:

- **Consumer subscription: $14.99/mo, ~$149–179/yr** — parity with or a slight undercut of SwingVision's $179.99/yr, justified by the L3 3D + coaching depth rather than discounted. A genuinely useful **free on-device tier** (L0/L1 advisory) is the acquisition funnel; the cloud 3D/biomechanics tier is the paywalled hook.
- **B2B seat-licensing:** academy/coach seats + college/HS team plans + a coach-facing dashboard (higher WTP, lower CAC, plus data rights).
- **Positioning vs SwingVision:** do **not** fight on line-calling latency/accuracy — publicly cede ELC. Headline the *3D world + grounded coaching* and *camera-motion tolerance*. Match SwingVision's table stakes (stats, clips, shot taxonomy) *before* differentiating on top.
- **Positioning vs Hawk-Eye:** never claim "officiating"/"official"/"ITF-certified" — those are governed terms (ITF Gold/Silver classification; USTA Reg III.C), and we are not classified. Badge every in/out as advisory. Document the ceiling honestly (Hawk-Eye ~3.6 mm / ~6–18 cameras vs SwingVision single-cam ~97% within 10 cm) so we never over-claim.
- **Unit-economics guardrail:** cap per-session cloud **L3 GPU COGS at <15% of monthly ARPU (~$1–3/deep session)**. This is structural: SwingVision's on-device inference is ~zero marginal cost, while our only gate-passing tier is cloud GPU (BODY = 96–98% of E2E). Un-optimized, a rally-gated match runs **~$11–29 in H100-spot time**; the warm-pool + batched-inference + Fast-SAM-3D-Body redesign (target **~$1.4–3.6/match**) is a prerequisite for viability, not a stretch. Reserve L3 for **user-selected serves/rallies**, never whole matches by default.

### E. Honest Threats

1. **SwingVision incumbency + data moat.** Apple Editors' Choice distribution, ~$10M raised ($6M Series A Oct-2023, led by Authentic Ventures with Sony Innovation Fund/GGV/Techstars), 200k+ players, and a **500M-shot** in-domain phone flywheel. It already added **pickleball** — it can come downmarket into any niche we open. (Note: **Sony**, Hawk-Eye's owner, is a SwingVision investor — an FTO consideration, not just a competitive one.)
2. **On-device COGS asymmetry.** Their ~zero marginal cost out-margins our cloud depth tier at every consumer price point; we must push more of the pipeline on-device (L0/L1) and shrink the paid server-deep surface.
3. **Line-calling is doubly commoditized** — Hawk-Eye at the top (tour standard, Wimbledon 2025), SwingVision + In/Out ($499) at the bottom. Our advisory-only stance *forfeits the category's #1 marketed feature*, so if 3D + coaching doesn't visibly land, we have no hook.
4. **"3D" is a contested marketing word** — SwingVision patents mesh deformation, Baseline claims 3D trajectory, Hawk-Eye ships skeletons — so our mesh is only a moat if it is *demonstrably, exportably* better than a 2D overlay.
5. **Fixed-camera rivals get better geometry for free**; our single moving camera makes tennis 3D materially harder (ball <10 px at the far baseline, 120–150 mph serves, heavy spin, multi-surface bounce), risking accuracy claims we can't back with a passed gate.
6. **Channel lock-out.** SportAI ($3M round Nov-2025, backed by Casper Ruud) is layering a **camera-agnostic API over MATCHi's 17k courts** — it could seize the club/coach channel before we enter B2B.
7. **The broadcast-data trap.** Tennis's huge corpus tempts de-funding the owner/user phone-capture flywheel that is the *actual* product bottleneck — SwingVision chose the phone flywheel over broadcast for a reason, and the platform's own 4×-measured in-domain-data lesson still binds. Broadcast footage is also **IP-encumbered** (broadcaster copyright + player publicity + Sportradar/TDI feed bans), so it is internal-pretrain/reference-only, never shippable weights.
8. **IP landmine.** SwingVision's granted **US 11,893,808 B2** (monocular NN 3D-property extraction on a mobile device) reads close to our Ball-3D + in/out pillar, and Infinity Cube's **US 10,467,478** was already litigated against SwingVision (S.D. Cal. 3:22-cv-00547) — a **freedom-to-operate opinion is a launch-blocking gate** before any commercial tennis release.

**Net:** tennis multiplies TAM, lifts ARPU, and opens institutional B2B — but the moat has to be the 3D world + grounded coaching + motion tolerance + honest trust, executed against a strong incumbent, funded with owner/academy in-domain capture, and cleared through FTO and no-officiating-claims gates before a dollar of paid acquisition.

# PART III — PHASE CHECKLISTS (agent-facing; every task self-contained)

Tennis `T`-prefix throughout. Every task names its parent pickleball task where it is a near-clone.
Evidence pointers: per-pillar research is `runs/research_tennis_20260707/<pillar>_report.md`; the code
delta is `runs/lanes/tennis_recon_20260707/`. `TENNIS-VERIFIED=0`; checkboxes mean "work item exists."
Bracket tags sequence work without renumbering. **Every shared-file change is gated on the pickleball
suite staying green (PART IV rule 15).**

## PHASE T0 — Sport-config foundation + tennis data engine (unblocks everything)

**Already built:** a genuine dual-sport court-geometry layer — `Sport = Literal["pickleball","tennis"]`,
a correct tennis `court_templates.py` entry (78×36 ft, 42″/36″ net, 21 ft service line, alleys),
`court_zones.py` tennis service-boxes+alleys, generic `net_plane.py`, `--sport {pickleball,tennis}`
flag, format-agnostic player roles (singles/doubles from track count). **To build:** wire the seam that
is currently unwired scaffolding; a tennis keypoint schema; a tennis physics stub; surface-tagged data
ingest + eval ledger; the first honest tennis world; the broadcast data engine; the profile registry;
the match-scoring FSM.

- [ ] **T0-1 [START HERE] Tennis court keypoint schema + fix the tennis stubs.** Add
  `TENNIS_COURT_KEYPOINT_NAMES` (~14–16: 4 doubles corners, 4 singles-baseline corners, 4 service-box
  outer corners, 2 center-service-line "T" junctions, 2 baseline center marks + net-top points; extra
  coplanar points *improve* single-view PnP identifiability); make `schemas/__init__.py`
  `per_keypoint_residual_px` length-validation sport-conditional (today hardcoded to len-15 regardless
  of `sport`); fix `court_corner_review.py:164` (tennis branch returns an empty `required_line_ids=()`,
  contradicting `court_line_evidence.py`'s tennis service-line requirement). Gate: `process_video
  --sport tennis` runs calibration end-to-end without the pickleball-15 assertion; both sports' schema
  tests green. Kill: none — pure code.
- [ ] **T0-2 [START HERE] Thread `StageContext.sport` end-to-end.** Wire `ball_arc_solver.court_sport`
  (defaults `"pickleball"`, never wired from the orchestrator today) and the ~12 backend sites that
  silently default `sport="pickleball"` (`ball_line_calls`, `ball_manual_court_inout`,
  `court_calibration_metric15`, `person_court_membership`, `raw_pool_/offline_person_authority`, …);
  add an assert-sport guard at every stage entry. Gate: a `--sport tennis` run computes tennis geometry
  at every stage (grep proof: no silent pickleball default reached); pickleball suite green. Kill: none.
- [ ] **T0-3 Tennis `PhysicsParameters` + ball profile stub.** Add a tennis ball profile (dia ~0.0657 m,
  mass ~0.058 kg; banded Cd seeded ~0.507; Cl(S)≈0.5–0.6·S; altitude-aware ρ) and **raise
  `selection_max_speed_mps`/`max_plausible_speed_mps` from 35 to ≥70** — a correctness bug: serves at
  54–67 m/s are silently rejected today. Real constants tuned in T1-4. Gate: solver accepts a 60 m/s
  serve arc and round-trips a synthetic tennis trajectory. Kill: none.
- [ ] **T0-4 Tennis data ingest + surface/format/domain tagging.** Tag each clip **surface
  (hard/clay/grass) × format (singles/doubles) × domain (broadcast/our-camera) × role
  (train/internal-val/held-out)** AT INGEST; wire every tennis clip ID + any partnership/broadcast/
  sensor GT source into `eval_guard.assert_not_training_on_eval_clip` on **day one** (more real GT =
  more leak surfaces). Gate: a tennis clip ingests fully tagged; eval_guard blocks training on a
  held-out tennis clip. Kill: none.
- [ ] **T0-5 Tennis eval-suite + pre-registered held-out ledger.** An our-camera held-out set
  stratified by surface × format × court-half; reuse `heldout_eval_ledger.md` discipline (pre-commit
  candidate+threshold from non-held-out evidence, exactly one scoring run, `append_lock`); a
  `PREREGISTRATION.md` per gate. Gate: ≥1 held-out row per available surface registered before any
  tuning. Kill: none.
- [ ] **T0-6 First honest tennis E2E world (TM1).** `process_video --sport tennis` on one broadcast + one
  owner clip, everything trust-banded. Gate: complete bundle; tennis court/net rendered; players
  placed; ball attempted; `verify_process_video_viewer.py` assertion_errors==[]; pickleball suite green.
- [ ] **T0-7 Broadcast data-engine spike (NEW pillar).** Distinct ingestion subsystem (no ARKit sidecar,
  moving PTZ camera): YouTube ingest → rally/shot segmentation → auto-homography court fit → WASB/
  TrackNet ball → trajectory-kink+audio bounce → scoreboard/serve-speed OCR. Footage never
  redistributed; models kept off any NonCommercial/broadcast-copyright path. Reference corpus:
  TennisVL/TennisExpert (202 matches, 471.9 h, 40,523 rally clips). Gate: N broadcast matches ingested
  with auto-homography + auto-ball-labels + a corpus card; the **broadcast→phone domain-gap probe** row
  registered (T0/T1). Kill: gap exceeds threshold → broadcast becomes physics/geometry/reference-only
  for detectors (still funds owner capture — SwingVision's moat is ~500M *phone* shots, not broadcast).
- [ ] **T0-8 Tennis profile registry.** Extend the registry for tennis: per-court (surface, tape-measured
  net heights, corners, lens intrinsics), per-racket (dims/scan), per-player (NTRP/UTR, handedness,
  1H/2H backhand). Gate: a tennis profile persists + re-identifies on upload. (mirrors P0-9)
- [ ] **T0-9 Match-scoring FSM (NEW pillar — the one pillar VERIFIABLE without ML).** Greenfield
  deterministic FSM (grep finds zero rally/score/deuce/tiebreak state today): points→game→set→match,
  deuce/advantage, no-ad, tiebreak/match-tiebreak (Coman rotation), config-driven Appendix VI variants;
  score-driven diagonal service-box validity; singles/doubles sideline switch; serve fault→2nd→
  double-fault sub-FSM; refuse intent-based calls (hindrance/let stay advisory). Gate: **100% exact** on
  a scripted point suite (deuce/no-ad/tiebreak/Coman/best-of-3/5) + validated vs Match-Charting-Project
  scorelines. Kill: none.

## PHASE T1 — Ball: transfer, true 3D flight, spin (the cheap-win pillar)

**Already built:** WASB-SBDT (MIT) + blurball + TrackNetV3 union candidate pool; `benchmark_ball_
trackers.py`/`ball_benchmark.py` harness (F1@Npx, teleport, max-jump); SST + CVAT loop; event-anchored
per-segment BVP arc solver (RK4 drag+gravity+Magnus); synthetic-lift (TT4D/UpliftingTT) pattern; a
`spin_rpm` schema slot (hardcoded `None`); trust bands. **To build:** checkpoint swap (tennis is
in-domain!), the serve-speed regime, felt-ball physics, spin, tiled far-ball, learned bounce/in-out.

- [ ] **T1-1 [START HERE — zero data] Ball transfer eval.** Load upstream tennis WASB/TrackNetV3
  checkpoints (both zoos ship them in `MODEL_ZOO.md`); score **zero-shot** on owner tennis clips + a
  broadcast set; measure the gap to bar AND the broadcast→phone drop. This is the highest-information
  cheap experiment in the whole program — it decides whether tennis ball tracking is nearly-free. Gate:
  transfer bet quantified; broadcast-view F1 ≥ 0.92 expected; our-camera F1 baseline recorded in the
  ledger. Kill: our-camera F1 collapses like pickleball → escalate to the full owner-capture data
  engine, re-scope T1.
- [ ] **T1-2 Tennis fine-tune — owner finisher, broadcast primer (only if T1-1 insufficient).** TOTNet
  occlusion recipe + blurball + a modest amateur low-angle iPhone set (100s–1000s frames) + SST;
  broadcast tuning is **REFERENCE-VERIFIED only** — a broadcast F1 is not the product number (the
  Roboflow-pickleball inversion lesson). Gate: our-camera held-out F1 ≥ 0.85, hidden-FP ≤ 0.05. Kill:
  broadcast-only fine-tune regresses our-camera → owner data only. (mirrors P1-1)
- [ ] **T1-3 Serve-regime recall (correctness bug, not tuning).** Re-derive the teleport/max-jump gate
  (current 160 px/frame + 3-frame gap **rejects every real serve** — a 54–67 m/s ball moves ~0.9–1.1 m/
  frame @60 fps, ~13 ball-diameters); make **≥120–240 fps the default for serve-containing rallies**;
  promote BlurBall-style streak modeling (position + blur-orientation θ + half-length ℓ; blur length is
  a free per-frame velocity cue, MAE 1.2 px vs WASB 3.1 px); retire TrackNetV4 from the live path (F1
  0.9581 < V2 0.9677, undeserializable weights). Gate: serve ball tracked through contact on 240 fps
  clips; serve recall ≥ bar. Kill: none.
- [ ] **T1-4 True tennis 3D flight (felt-ball physics).** Reseed Cd to a **banded, Reynolds/wear-
  dependent** prior (~0.507 Goodwill/Cross new-ball mean, contested vs Mehta 0.6–0.7; fuzz ≈10% of Cd,
  wear ≈6% — the old "up to 40% / toward 0.4" figures are dropped); Cl(S)≈0.5–0.6·S with a small
  intercept, S up to ~0.45; altitude-aware ρ. Serve-speed + spin-axis become first-class (drag ~5 g at
  serve speed makes the arc strongly observable). Gate: 3D-arc reprojection ≤ detector noise;
  serve-speed ≤ 5% / ±3 mph below 100 mph vs radar GT; degraded band above. Kill: none. (mirrors P1-4)
- [ ] **T1-5 Spin estimate (NEW pillar — `spin_rpm` is `None` today).** Ship **CLASS first**
  (topspin/backspin/slice/flat) via a trajectory-inverse transformer trained on tennis MuJoCo synthetics
  (retarget UpliftingTableTennis, Kienzle et al. WACV 2026, **GPL-3.0 → clean-room reimpl, never ship
  weights**); then a coarse 3-band RPM. Precise serve RPM only @240 fps + radar GT; **full 3D spin-axis
  is NOT gated** (even an event camera errs ~33° on axis). Honest ceiling: the "97% spin" headline is
  *binary* topspin-vs-backspin on synthetic (97.1%), 89.5% on real captured TT footage. Gate:
  spin-class macro-F1 ≥ 0.85; coarse RPM band ≥ 80% vs radar. Kill: RPM MAE exceeds band → class-only,
  RPM advisory. (mirrors P1-5)
- [ ] **T1-6 Learned bounce/contact events + surface-aware.** Retune audio for the strung-racket ~5 ms
  "thwock" (vs table-tennis ~1.3–1.8 ms, vs pickleball pock) — bandpass/min-separation/HFC + hit-surface
  classification (racket/ground/net-cord); TTNet-style learned event head over track+audio+wrist;
  surface-conditioned post-bounce state; clay ball-mark bounce anchor (see T4-6). Gate: contact timing
  p90 ≤ 40 ms; bounce-vs-hit ≥ 0.9 F1 internal-val, held-out pre-registered. Kill: none. (mirrors P1-6)
- [ ] **T1-7 In/out with modeled uncertainty (advisory — cede ELC).** Swap `PICKLEBALL_DROP_TEST_HEIGHT_M`
  (1.98 m) for the ITF rebound spec (254 cm drop → 135–147 cm); framerate-gate confident serve calls
  (≥120–240 fps — at 30 fps a serve moves ~1.8–2.2 m/frame, inflating σ_depth to meters, **the dominant
  tennis failure mode**); add service-box/singles-vs-doubles line context; σ_bounce + `too_close_to_call`.
  Gate: clear-call agreement ≥ 0.95, **zero confident-wrong**, every call σ-banded; advisory-not-
  officiating copy. Kill: none. (mirrors P1-7)
- [ ] **T1-8 Ball trajectory forecasting.** Cross-attention forecaster (racket K/V, ball Q) for
  occlusion-bridging (serve/net) + coaching anticipation. Gate: bridges occluded segments; improves
  recall. (mirrors P1-8)

## PHASE T2 — Body: tennis-dynamic hardening + the FIRST real BODY gate

**Already built:** frozen SAM-3D-Body + MHR70 (a generic anatomical rig — a tennis player is a human);
classical world-grounding (person-masked LK+MAD, foot-plane, MAD+Gaussian/MHR-latent smoothing — the
SMART/WorldPose-winning stack); YOLO26m+BoT-SORT+OSNet; SOMA-X SMPL-X interop; challenger bench
protocol. **The genuinely good news:** unlike pickleball (BODY *never* GT-validated), tennis has real
3D benchmarks (CalTennis, AthletePose3D) — this pillar can earn its **first VERIFIED body gate**.

- [ ] **T2-1 [START HERE] Phase-gate OFF foot-lock during serve flight.** Both feet airborne at full
  extension → the foot-pin (the single most load-bearing grounding trick) drags the airborne body down.
  Add a per-foot airborne/contact classifier gating the foot residual. **Highest-severity grounding bug
  tennis introduces.** Gate: 0 phantom foot-pins on jump-serve clips; world foot-pos ≤ 0.30 m. Kill: none.
- [ ] **T2-2 Per-phase adaptive MHR-latent smoothing.** Near-zero window across the serve acceleration
  snap (shoulder IR ~2400°/s, wrist ~1900°/s), W=9 elsewhere — the documented SMART over-smoothing
  failure, amplified; make the wrist-latent hybrid default. Gate: serve snap preserved (angular-vel not
  flattened) AND jitter elsewhere ≤ bar. Kill: none. (mirrors P2-2)
- [ ] **T2-3 Axial limb rotation as a separately-trust-banded output (research-open).** Humeral IR/ER +
  forearm pron/sup drive ~41%/~32% of racket velocity and are exactly the DOF monocular HMR is weakest
  at; **likely UNVERIFIED until T3 racket-6DOF fusion — do NOT ship as measured.** Gate: banded output
  present; no measured claim without T3 fusion + a validated gate.
- [ ] **T2-4 Racket-occlusion + two-handed-backhand contact model.** PromptHMR-style mask/box prompts for
  racket occlusion; a dual-hand contact model for the 2H backhand. Gate: reduced limb dropout under
  racket occlusion.
- [ ] **T2-5 Court-anchored scale/translation + far-player crop.** Monocular translation error is 0.9–3.6 m
  at 13–17 m capture distance (CalTennis) → lean on the court plane + ARKit sidecar (LiDAR range ~5 m ≪
  24 m court); high-res crop re-inference for the far-baseline player. Gate: far-player recall ≥ 90% at
  ≥ 24 px box; world foot-pos ≤ 0.30 m. (mirrors P2-3)
- [ ] **T2-6 First independent BODY GT gate (the project's first-ever).** Register held-out rows; run the
  frozen SAM-3D-Body + grounding vs **CalTennis** multi-view-consistency (CC-BY-NC — eval/pretrain
  only; a *lower bound*, cross-check vs marker GT) + **AthletePose3D** (general 12-sport, NC — its
  214→65 mm gain is NOT tennis validation) + owner marker serve mocap; quantify the proximal-vs-distal
  accuracy envelope before committing thresholds. Gate: PA-MPJPE ≤ 70–85 mm; world foot-pos ≤ 0.30 m
  (beat PromptHMR's 0.94 m). Kill: none. (mirrors P2-6)
- [ ] **T2-7 Challenger re-benchmark on tennis footage.** PromptHMR (best CalTennis translation 0.942 m),
  KASportsFormer (SportsPose 58 mm / WorldPose 34.3 mm, skeleton-only), Human3R vs the hardened SAM-3D
  stack; pre-registered decision rule (beat on ≥3/4 of {world-MPJPE, jitter, foot-slide, far-player} by
  ≥20%). Kill: no challenger clears → keep SAM-3D. (mirrors P2-7)

## PHASE T3 — Racket 6-DOF (paddle → racket)

**Already built:** `paddle_pose_fused` end-to-end (MHR70 hand-frame H(t), constant grip-transform G via
weighted Wahba, ball-reflection face-normal evidence, wrist-gated box correction, per-segment grip-roll
IoU search); **IPPE planar-PnP still holds** — a strung face is a rigid planar ellipse with a
well-defined normal; the footprint is *already* modeled as an ellipse; SLERP one-euro rotation
smoothing; our 5-keypoint schema ≈ RacketVision. **To build:** racket geometry, long-lever handling,
grip taxonomy, 2H backhand, swing-speed via lever, sensor GT.

- [ ] **T3-1 [START HERE] Strung-face geometry + detection.** Swap the 16″×8″ rectangle / 5.25″ handle for
  a **27″ (0.686 m) racket, elliptical head ~318×265 mm, long throat+grip**; retrain the detector on the
  RacketVision RTMPose-M seed (MIT, PCK@0.2 89.6%) → owner labels; **explicitly lift the weak face-width
  (left/right) keypoints (~80% vs >92% structural)** — they set the face-normal roll. Gate: racket PCK@0.2
  ≥ 0.85 overall **and ≥ 0.80 on face-width**. Kill: strung-face silhouette too weak → hand-frame-only
  pose, defer face-normal. (mirrors P3-1/P3-4)
- [ ] **T3-2 Long-lever pose + jitter.** The longer lever t_g (~3–4× the paddle) amplifies small H(t)
  rotational noise into large face-tip excursion; re-tune the closed-form t_g least-squares and make the
  lever explicit; SLERP one-euro (needed more than the paddle). Gate: face-normal jitter ≤ 5°/frame.
- [ ] **T3-3 Tennis grip-prior library + 2H backhand.** Eastern/Semi-Western/Western/Continental/serve-
  pronation replacing the single continental prior (per-segment-constant-G is *weaker* for tennis — serve
  pronation spins the face fast within one contact); a **dual-wrist grip model** for the 2H backhand.
  Gate: grip-consistent pose across stroke types.
- [ ] **T3-4 Swing-speed via lever arm.** `v_head = v_wrist + ω×r_lever` (racket head 30–45 m/s at serve,
  far above wrist speed — wrist displacement badly under-reads). Gate: swing-speed vs sensor-racket GT
  (target ICC ~0.98). (feeds TS)
- [ ] **T3-5 [FAST-TRACK] Ball-impact inversion (activate when T1-4 3D velocities land).** The dormant
  factor; **bounded at 26.4±4.4° (TT4D mocap) → stays advisory** on its own. Gate: face-normal ≤ 15°
  only WITH keypoints+PnP; inversion-alone stays advisory. (mirrors P3-5)
- [ ] **T3-6 Contact-point-on-string-bed + blur-axis spike.** Model string-bed give ~10–30 mm (the rigid
  37 mm impact cap is wrong); a blur-axis-as-orientation channel for the near-transparent strung face on
  fast swings (novel, publishable). Gate: contact point localized on the face; blur channel A/B.
- [ ] **T3-7 Sensor-racket GT + owner marker GT (→ RKT tennis-VERIFIED).** A legacy Zepp/Babolat unit
  (category EOL 2020–2021 — **source second-hand now**) for swing-speed GT (ICC 0.983); 4-marker owner
  clips for face-angle GT. **Do NOT use sensor impact-location (κ 0.217) as a gate.** Gate: face-angle
  ≤ target p90 vs marker GT; swing-speed vs sensor. Kill: none. (mirrors P3-7)

## PHASE T4 — Court/net + multi-surface

**Already built:** the correct tennis `court_templates.py` entry + `court_zones.py` service-boxes/alleys
+ `net_plane` API + metric-15pt PnP (8 px/15 px gates) + the LM point+line seam + ChArUco k1/k2 +
`CourtProfile` registry + the heatmap→homography→reference-snap pattern + the `OVERLAPPING_COURT_
CALIBRATION` HSV/Hough/shadow stack (already tennis-shaped). **To build:** the tennis keypoint detector,
a singles/doubles classifier, a piecewise net, and the multi-surface pillar.

- [ ] **T4-0 [START HERE] Tennis court profiles (owner courts).** Store a frozen calibration + line color
  + lens intrinsics per owner court+surface; re-identify on upload (fingerprint + color ΔE + 4-line
  reproj). Gate: profile round-trip reuse on the owner's courts. (mirrors P4-0)
- [ ] **T4-1 Land the tennis template end-to-end + singles/doubles classifier.** Wire the tennis 14–16 pt
  schema (service boxes, T-junctions, singles sidelines — no kitchen); build a **runtime singles/doubles
  classifier** (the same 36 ft lines serve both formats → format cannot be read off geometry; fuse
  person-count 2 vs 4, lateral positions, serve-landing box). A wrong `court_mode` silently corrupts
  in/out AND zones AND the net model. Gate: court renders correctly singles+doubles; format classified.
  Kill: none.
- [ ] **T4-2 [UNKNOWN-COURT EPIC] Tennis court keypoint detector (rehabilitates the killed architecture).**
  Reimplement the heatmap-then-points CNN our own `court_unet_v2` KILLED (ledger rows 70–72: PCK@5
  0.017–0.056) — the failure was pickleball data scarcity, now viable with 8,841 public tennis labels.
  **Do NOT ship yastrebksv weights** (no license → all-rights-reserved; the Gholamreza HF "MIT" mirror is
  an invalid re-upload of the same broadcast frames) — reimplement under our license, pretrain broadcast
  → fine-tune our-camera; its headline 96.3%/1.83 px is the *full* refine+homography config (raw base
  93.6%/2.83 px). Gate: PCK@5px ≥ 0.95 per surface on our-camera held-out; surveyed-corner reproj ≤ 5 cm.
  Kill: our-camera PCK collapses → metric-15pt manual path stays primary. (mirrors P4-2)
- [ ] **T4-3 Piecewise net model.** Generalize `net_top_height_m_at_x` from a single linear post→center
  ramp to a **piecewise profile** — center strap pinned to 0.914 m, rising to 1.07 m at doubles posts OR
  at **singles sticks (±(27/2+3) ft) with the outboard alley sagging below** (a genuine double-dip the
  linear model cannot represent). Gate: net height ≤ 2 cm vs tape-measured at posts+center+sticks.
  (mirrors P4-6)
- [ ] **T4-4 Multi-surface court detection (NEW pillar).** Surface classifier (clay red-orange / hard
  blue-green-gray / grass green) + the court-color-context filter (the 7×7 "neighbors match court color
  but pixel does not" rule from the amateur-court paper, Agrawal–Sundararajan–Sagar 2404.06977) + shadow
  preprocessing. Gate: surface classify ≥ 97%; PCK holds on clay/grass low-contrast held-out.
- [ ] **T4-5 Surface-indexed bounce physics (NEW pillar).** Per-surface {COR e_v, friction μ,
  spin-coupling}: clay e≈0.83/μ≈0.8 (slow-but-high), hard e≈0.80/μ≈0.7, grass e≈0.73/μ≈0.6 (fast-low);
  adopt the **ITF Court Pace Rating** CPR = 100(1−μ) + 150(0.81−e_T) as the surface knob (distinct from
  Hawk-Eye's in-match CPI); Cross grip/slide regime (slide on grass/shallow angles, grip/kick on clay
  with topspin). Gate: per-surface bounce-height ≤ 10% vs GT.
- [ ] **T4-6 Clay ball-mark review (NEW sub-pillar — the flagship surface differentiator).** Segment the
  physical skid mark; fuse with the trajectory-derived bounce for a mark-anchored **advisory** in/out +
  skid-length (a spin/angle proxy) — a physical ground-truth mechanism with **zero analog in pickleball
  or on any other surface.** Advisory only (a single non-ground camera cannot match Foxtenn's ~40-camera
  clay rig). Gate: clay ball-mark advisory ≥ 85% agreement (explicitly NOT vs Hawk-Eye). Kill: mark
  segmentation unreliable → drop to a "see your bounce" visual only.
- [ ] **T4-7 Capture defaults for the bigger court.** Default tennis to **4K60** (far-baseline ball ~4 px
  @1080p → ~8 px @4K at ~24 m); elevated behind-baseline mount centered on the center mark; downgrade
  LiDAR/ARKit-plane grounding (LiDAR ~5 m ≪ 24 m) — rest on line-homography. Gate: far-ball pixel floor
  cleared on 4K.

## PHASE TS — Serve biomechanics (NEW flagship pillar)

**Already built:** the 3-stage coaching pipeline maps 1:1 onto serve coaching; "serve" is already a
`SHOT_TYPE`; `contact_windows` fuses wrist-velocity + ball-inflection + audio into a contact instant
with trust bands; the iOS sidecar's **240 fps + ARKit pose + gravity + LiDAR + locked exposure** ships
(the single most important enabler); the ball zoo + arc solver (toss/serve-speed); the racket scaffold.
**To build:** the 8-stage segmenter, the reference library, the 240 fps mandate, the distal-advisory
ring-fence, foot-fault, and non-clinical injury markers.

- [ ] **TS-1 [START HERE — pose-only, no new data] 8-stage serve segmenter.** Kovacs–Ellenbecker stages
  (start / toss-release / loading / cocking-trophy / acceleration / contact / deceleration / finish)
  anchored on the fused contact instant; gates foot-lock (T2-1), adaptive smoothing (T2-2), and feature
  windows. Gate: contact-frame ≤ 1 frame @240 fps; kinematic-sequence ORDER ≥ 90%. Kill: none.
- [ ] **TS-2 Serve reference-range library (signed).** Seed from Kovacs 8-stage + Elliott ITF: max ER
  172±12° reached ~0.09 s pre-contact, front-knee flexion 24±14° at contact, trunk tilt 48±7°, contact
  racket velocity 38–47 m/s, peak vertical GRF 1.68–2.12 BW, legs+trunk 51–55% of energy; racket-head
  velocity = 54.2% shoulder-IR + 31.0% wrist-flexion + 12.9% horiz. Each range signed with a citation +
  owner-review flag. Gate: `validate_reference_ranges.py` green; coach `signed_off_by`. (mirrors P6-3)
- [ ] **TS-3 240 fps mandate + proximal metrics.** **Refuse / heavily-band 30–60 fps serves** — the
  acceleration phase (max-ER → contact) lasts **< 0.01 s**, unsampleable at 60 fps (16.7 ms/frame). Ship
  proximal metrics (leg drive, knee flexion, hip/trunk rotation, trophy statics, contact height,
  sequence ordering) where markerless validates (OpenCap RMSE 2.0–10.2°; HRNet mechanical-work within
  7.3/9.3 J). Gate: proximal angular-velocity within ~10–15% vs OpenCap/mocap.
- [ ] **TS-4 Serve-speed (radar GT) + foot-fault advisory.** Homography + drag serve-speed (degrade
  > 100 mph); a **precision-first** foot-fault advisory at the baseline (Hawk-Eye dedicates 6 close
  cameras to foot faults → a single far phone is an advisory ceiling). Gate: serve-speed ≤ 3 mph / 5 km/h
  MAE vs radar (radar itself ±1 km/h); foot-fault precision-first.
- [ ] **TS-5 Distal velocity / racket-head-speed — RING-FENCED advisory (research-open).** Shoulder IR
  ~2420°/s + wrist ~1950°/s is the platform's most-wanted serve number and its **least-reliable
  markerless signal** — out-of-scope / trust-banded until the T3 racket retool + a validated gate.
  Injury markers ship as **non-clinical kinematic proxies only** — video cannot measure ~0.5–0.75 BW
  shoulder distraction or ~300 N·m deceleration torque, and stating them as load is a medical-liability
  trap. **NEVER shipped as measured.** Gate: no measured distal claim without T3 fusion + a passed gate.
- [ ] **TS-6 Serve coaching card.** Compose proximal metrics + kinematic-sequence + serve speed +
  foot-fault into a grounded, fabrication-audited serve card (the highest-value coachable unit in the
  sport). Gate: 0-fabrication; ≥3 finding types; jump-to 3D moments. (feeds T6)

## PHASE TF — Global fusion: one mutually-consistent metric tennis world (at tennis speed)

**Already built:** PF-1 bounded confidence-gated post-hoc nudges + PF-2 whole-clip coordinate-descent
(block-sparse scipy `least_squares` trf+huber, existing confidence fields as residual weights);
residual families (ball↔racket impact, ball↔ground bounce, foot↔ground, hand↔grip, net-height, scale);
F0 read-only meter + F4 `assert_cross_system_consistency`; the ARKit-sidecar camera-lock; JOSH as the
reference pattern. **What breaks is the SPEED/PHYSICS regime, not the architecture.**

- [ ] **TF-1 Cheap consistency priors first.** Ball↔**strung-face** impact snap + foot↔ground clamp
  (airborne-gated, T2-1) + non-penetration. Gate: impact-gap + floor-penetration strictly decrease, zero
  regression on standalone metrics. (mirrors PF-1)
- [ ] **TF-2 Contact-coupled joint optimizer at tennis speed.** **≥120 (prefer 240) fps precondition** —
  a 200 km/h serve moves ~1.85 m/frame at 30 fps; below 120 fps arc-association and contact-localization
  are ill-posed (multiplies the variable/residual vector 4–8× → **compute, not modeling, becomes the
  binding constraint**). Continuous sub-frame `t_contact` per impact (dwell ~4–5 ms < one 120 fps frame
  → **promote audio onset (sub-ms) to the primary contact cue**); spinning-ball Cd(Re)+spin free
  variables; elliptical strung-face impact with string-bed give ~10–30 mm; dual-wrist grip on the 27″
  lever. Gate: beats PF-1 on `impact_gap_max_m` ≤ 0.05 + penetration; consistency asserted. (mirrors PF-2)
- [ ] **TF-3 Surface-coupled bounce residual + clay ball-mark anchor.** Gate: bounce residual
  surface-conditioned; clay-mark anchors the bounce where available.
- [ ] **TF-4 Compute containment (research-open).** JOSH runs ~0.8 FPS on a 4090 → a 2400-frame 240 fps
  clip is naively out of the offline budget. Adaptive temporal resolution (full fps only in
  contact/bounce windows), a B-spline/low-rank temporal basis for camera + body-root, JOSH3R feed-forward
  init, H100. Gate: match-scale fusion within the cost envelope (T5). Kill: intractable → fusion only on
  user-selected serves/rallies, never whole matches.
- [ ] **TF-5 Broadcast-camera fusion path.** Per-frame 6-DoF from court-line PnP (no ARKit sidecar),
  degraded trust band, range-aware far-player weighting (13–17 m). Gate: a broadcast clip fuses with
  honest degraded bands.

## PHASE TL — Live tier (on-device advisory; parallel stream, never blocks the critical path)

**Already built:** `LiveFrameTap`, the ANE benchmark harness, the L0/L1/L2/L3 constitution, person
track (2/4 players), the 240 fps capture sidecar, the WASB temporal family, the coming-soon
kill-switch. **To build:** decouple ball from the cadence scheduler, a tennis ball student, a tennis
court lock + singles/doubles, advisory calls + serve speed first-class, the thermal soak.

- [ ] **TL-1 Decouple ball from the cadence scheduler.** Every-4th-frame = ~4.5 m gaps at 150 mph
  (fatal) — ball must run every frame @60 fps live. Gate: 0 skipped ball @60 fps; live-loop p90 <
  16.6 ms/frame. Kill: budget blown → cap live ball to slower shots, band fast serves.
- [ ] **TL-2 Live tennis court lock + singles/doubles toggle.** Guided corner tap + ARKit plane assist +
  profile reuse; tennis geometry + format toggle. Gate: reprojection p95 ≤ our manual bar. (mirrors PL-1)
- [ ] **TL-3 Tennis ball student distillation (product-gated on T1).** Distill server WASB → a CoreML
  tennis student after T1 clears bar; resolve the 960 px ANE-compile failure vs. far-ball resolution.
  Gate: on-device F1 within band of server; ANE latency budget met. (mirrors PL-5)
- [ ] **TL-4 Advisory line-call + serve-speed first-class (ELC-ceiling honesty).** Gate: advisory copy
  only; σ-banded; never "officiating." (mirrors PL-4)
- [ ] **TL-5 Record+infer soak benchmark.** ≥90-min outdoor thermal soak — the "217 fps headroom" is a
  mirage (excludes the real temporal tracker/pose/render/thermal; the partial loop is already 4.59 ms >
  the 4.17 ms budget). Gate: `video_frame_drop_rate` ≤ 0.005 over 90 min outdoor. Kill: thermal fails →
  cadence back-off; the record path stays sacred. (mirrors PL-2)
- [ ] **TL-6 Server fast-verdict L2 tennis + live serve/stroke overlays.** A `process_video` profile
  skipping BODY for a ~1–2 min calls+stats fast path; surface the built live guidance/preview cards.
  (mirrors PL-6/PL-3)

## PHASE T5 — Speed + cost + match scale

**Already built:** TensorRT/FP16 (WASB+YOLO only), the e-process sequential-hypothesis auto-QA, the
warm-pool/batched-BODY levers, the Render service, fully-loaded cost metering. **To build:** match-scale
orchestration (the new tennis axis).

- [ ] **T5-1 [PREREQUISITE, not a stretch] Match-scale orchestration.** A match-level **rally splitter**
  in front of the per-clip pipeline (a best-of-5 match is up to ~1.08M frames but only ~10–25% ball-in-
  play); client-side rally-gated upload (a raw 4K60 match = 14–55 GB); per-rally QA isolation +
  checkpoint/resume; point boundaries via score OCR (T0-9 FSM). Gate: a match processes rally-gated
  within ≤ 2× play duration.
- [ ] **T5-2 Mandatory warm-pool batched BODY inference.** Un-optimized, a rally-gated match runs ~$11–29
  in H100-spot time; target ~$1.4–3.6/match (Fast-SAM-3D-Body, a training-free ~10.9× acceleration —
  re-verify checkpoint parity before trusting the speedup in an L3 gate). **Reserve L3 for user-selected
  serves/rallies, never whole matches by default.** Gate: deep-session L3 COGS < 15% of ARPU (~$1–3).
  Kill: can't hit the envelope → L3 opt-in per-rally only. (mirrors P5-7)
- [ ] **T5-3 Higher-fps/VFR ingest + per-match cost metering + surface-stratified auto-QA.** (mirrors
  P5-4/P5-6)

## PHASE T6 — Coaching + product output (the end goal)

**Already built (transfers better than any other pillar — its method was borrowed FROM tennis):** the
entire 3-stage grounded-LLM (deterministic features → non-LLM reference-range comparator → format-locked
LLM that never sees a raw number), the `reference_ranges` schema + `validate_reference_ranges.py`, the
`shot_rules.py` evaluator, the fabrication firewall (citation cross-check + 0/300 `audit_coaching_
fabrication.py`), abstention discipline. **Only the ROWS and taxonomy change.**

- [ ] **T6-1 Tennis stroke recognition (replace the ball-geometry classifier).** pickleball's
  `_classify_shot_type` is trajectory-only — it *physically cannot* see FH vs BH, 1H vs 2H backhand, or
  topspin vs slice (pose/swing phenomena). Build a NEW **pose+trajectory** pillar (port the BST
  cross-attention over MHR skeletons + arc; PoseC3D backbone, Apache-2.0). Tennis taxonomy: serve
  (flat/kick/slice), FH & BH (topspin/slice/flat), 1H/2H backhand, volley, half-volley, overhead, drop,
  lob, approach, return; contact_zone → baseline/no-man's-land/service-box/net; infer topspin vs slice
  from swing path (low-to-high vs high-to-low) + bounce (spin is sub-frame at 30–60 fps; 240 fps helps).
  Gate on **macro-F1** (serve-class rarity): coarse 4-class ≥ 0.85; 1H-vs-2H ≥ 0.80; topspin/slice ≥
  0.70; beat THETIS 79.17% with 3D features. Kill: pose too noisy far-court → coarse classes only.
  (mirrors P6-1)
- [ ] **T6-2 Tennis stat layer.** First/second-serve %, 1st/2nd-serve points won, ace/DF rate, winners &
  unforced errors + W:UE, break-point conversion/saved, net-point win %, return-points won, rally-length
  distribution; delete third-shot-drop/dink/kitchen/two-bounce. Gate: stats match hand-charted on
  held-out. (mirrors P6-2)
- [ ] **T6-3 Tennis reference-range library (the moat) — UTR/NTRP.** Seed from the **Match Charting
  Project** (17,808 matches / 10.5M shots) + ATP/WTA — but these are **CC-BY-NC-SA + pro-only**:
  facet-flag every row `pro_level`, **derive only aggregate statistics into a clean-room library, never
  redistribute rows**, and pair with an owner longitudinal self-baseline. Skill axis = **UTR
  (1.00–16.50) primary + NTRP coarse map + WTN note** (not pickleball's 3.0/4.0). **Self-relative framing
  is the default voice** (comparing a rec 3.5 to Djokovic distributions is invalid and demoralizing).
  Gate: reference library signed + provenance-tiered; validator green. Kill: rec-level distributions
  unavailable → self-relative only. (mirrors P6-3)
- [ ] **T6-4 Grounded-LLM tennis coach + automated UE with an explicit trust band.** UE is human-subjective
  (even pro charters disagree) — validate agreement vs MCP human UE labels; if low, demote to "error
  under low pressure." Gate: 0/300 fabrication audit; coach rubric ≥ 8/10; every stat trust-banded.
  (mirrors P6-4)
- [ ] **T6-5 Surface facet + serve breakdown (from TS-6) + visual feedback in the viewer.** Surface facet
  on ranges/features; jump-to-3D-moment overlays for each finding. (mirrors P6-5/P6-6)

## PHASE T7 — Productization (tennis mode)

**Already built (pickleball):** upload service, accounts, the H0 onboarding wizard, the VERIFIED-ladder
discipline. **To build:** the iOS tennis strategy, onboarding, pricing, and the launch-blocking legal
gates.

- [ ] **T7-0 [BLOCKING per PART 0] iOS sport strategy + a `Sport` selector that reaches the backend.**
  Every iOS target is `Pickleball*` and **no `Sport` concept exists in Swift**, while the backend
  defaults `sport="pickleball"` at ~12 sites. Decide parallel-mode vs. separate app (PART 0 ruling),
  then either rename/generalize or add a parallel module set + a `Sport` selector sent to the backend.
  Gate: iOS tennis mode selects sport end-to-end.
- [ ] **T7-1 Tennis onboarding + accounts + capture guidance.** Per-surface courts, racket, NTRP/UTR;
  on-device framing coaching for the bigger court (elevated behind-baseline; 240 fps serve prompt).
  (mirrors P7-1/P7-2)
- [ ] **T7-3 Pricing vs SwingVision/Hawk-Eye.** $149–179/yr (parity/slight undercut of SwingVision's
  $179.99), a genuinely useful free on-device L0/L1 funnel, the cloud 3D+coaching tier paywalled; a B2B
  academy/college seat engine (~$8.4B academy economy; SportAI is racing to lock MATCHi's 17k courts).
  **Never claim "officiating"/"ITF-certified"/"official"** (governed terms; we are not classified).
- [ ] **T7-4 [LAUNCH-BLOCKING] Legal / IP / FTO.** A written FTO opinion on **SwingVision US 11,893,808 B2**
  (monocular NN 3D-property extraction on a mobile device — reads close to our ball-3D + in/out pillar)
  and **Infinity Cube US 10,467,478** (already litigated vs SwingVision, S.D. Cal. 3:22-cv-00547; note
  Sony — Hawk-Eye's owner — is a SwingVision investor); **quarantine all NonCommercial/no-license
  training data** (CalTennis, MCP/ATP/WTA, AthletePose3D, TrackNet-tennis, TennisCourtDetector) from any
  shipped weights; a no-officiating-claims marketing audit. Gate: FTO cleared + quarantine audited
  **before any paid tennis acquisition**. Kill: FTO blocks → design-around or hold commercial launch
  (internal use unaffected). (mirrors P7-4)
- [ ] **T7-4b Data-privacy/retention (blocking before first non-owner tennis footage) + T7-5 tennis
  VERIFIED ladder.** Promote each pillar through its documented gate on our-camera labels; **market only
  OUR-CAMERA-VERIFIED capabilities**, never REFERENCE-VERIFIED ones. (mirrors P7-4b/P7-5)

# PART IV — STANDING RULES FOR EVERY AGENT ON THE TENNIS PROGRAM

These extend (do not replace) the pickleball `NORTH_STAR_ROADMAP.md` PART IV rules. Where a rule is
identical it says "inherited"; tennis-specific additions are called out.

1. **Read order for a fresh tennis session:** `CLAUDE.md` → `FABLE_OPERATING_MANUAL.md` → this file
   PART 0 (any blank owner item = typed STOP) + PART I (incl. I.7 critical path) → the parent
   `NORTH_STAR_ROADMAP.md` (the engine you are extending) → `BUILD_CHECKLIST.md` (last ~15 bullets) →
   `runs/manager/gpu_fleet.md` → `CAPABILITIES.md` (canonical truth) → the linked lane/run evidence.
2. **Protected data (tennis extends the pickleball policy):** a tennis capture/clip gets a role
   (train / internal-val / held-out) **and a surface tag (hard/clay/grass)** AT INGEST; held-out ones
   inherit full protection and are never touched without a pre-registered
   `runs/manager/heldout_eval_ledger.md` row + a manager STOP. **Surface discipline:** a model may not
   claim a surface it has not been eval'd on — a hard-court-only detector is `hard-court PREVIEW`, not
   tennis-VERIFIED. Broadcast-harvested clips are OUT-of-domain (their camera, not ours) → pretrain/
   diversity + broad-test only, never the in-domain finisher and never a held-out shot.
3. **Truth discipline (inherited):** VERIFIED requires the documented gate on real tennis labels.
   `TENNIS-VERIFIED=0` until earned; a passed pickleball gate does NOT transfer. Trust bands on every
   degraded/predicted/estimated output; fail-closed stays fail-closed; honest kills are wins. Live
   line/serve calls are ADVISORY forever — Hawk-Eye/ELC is the officiating gold standard and we do not
   claim it from one phone (PART III TS/T1 note; kill-list rule 5).
4. **Lane protocol (inherited):** file-disjoint concurrent lanes; explicit file ownership; coordination
   only via BUILD_CHECKLIST bullets + commit messages; artifacts under `runs/lanes/<lane>_<date>/`;
   wide blast-radius tests (`MPLBACKEND=Agg`); every new CLI ships its direct-CLI-reference test
   same-lane. **Tennis addition:** any lane touching a `sport`-parameterized file (`court_templates`,
   `ball_arc_solver`, `shot_taxonomy`, schemas) must run BOTH sports' tests — never regress pickleball
   while adding tennis.
5. **Kill list (inherited pickleball kills stay killed) + tennis-specific:** do NOT re-attempt without
   new evidence — officiating-grade single-camera line/foot-fault calls (Hawk-Eye/ELC uses 6–10+
   calibrated cameras; nobody does it from one consumer camera); FoundationPose-class zero-shot on the
   racket (HANDAL evidence, worse for a strung frame than a solid paddle); training a tennis model on
   competitor-processed video (SwingVision/Hawk-Eye outputs); treating broadcast-only fine-tunes as
   in-domain (the Roboflow-pickleball inversion lesson applies to broadcast-vs-phone tennis too);
   full 3D spin-vector recovery from a single view (unidentifiable — scalar top/back + sign only).
6. **License stance (inherited):** private use for now → licenses are not a constraint; keep a one-line
   "what we used" inventory per lane. Build nothing hardcoded to the owner or to one court/surface —
   sport/court/surface specificity lives in profiles with generic fallbacks.
7. **GPU FLEET (inherited):** safe-parallelism check before every lane (file/data/resource-disjoint);
   one physical GPU per lane; SPOT + teardown-on-completion; ≤$5/GPU/hr, max 4 concurrent; track in
   `runs/manager/gpu_fleet.md`. **Tennis note:** broadcast-scale training corpora are larger — budget
   higher per-wave spend explicitly; a 5th GPU or >$5/hr = `needs-purchase-approval` STOP.
8. **Known traps (inherited) + tennis-specific:** the `--sport` flag must be passed AND threaded — a
   silent `sport="pickleball"` default at ~12 backend call sites will make a "tennis" run compute
   pickleball geometry and pass no gate (T0 fixes this; until then, assert sport at every stage entry);
   tennis serves move the ball ~1–2 m between 60 fps frames — timing/association code tuned to
   pickleball's ~40 mph ball will drop serves (T1 fps/motion-model note); clay ball-marks and green
   grass change ball–background contrast (per-surface detector eval, rule 2).
9. **STOP-AND-ASK when genuinely blocked (inherited):** blocked = a decision needing info/judgment/
   money/authority ONLY the owner has AND no standing rule covers it. Classify into one bucket
   (needs-validation / needs-advice / needs-labeling / needs-decision / needs-purchase-approval) and
   surface it AS THE RESULT. Tennis's first-wave blockers are pre-listed in PART 0 (sport-config
   posture, surface scope, harvest ruling) — do not guess past them.
10. **Signal-adoption discipline (inherited):** before adopting any new vision signal, re-derive the
    current ablation from repo artifacts and run the 10-line pixel-math conditioning check at real
    tennis working distances (a bigger court means the far player/ball is smaller in-frame than
    pickleball — geometry that barely worked for pickleball may not clear the tennis baseline).
11. **Remote-code integrity (inherited):** BODY dispatch ships DATA never code; never trust a
    VM-computed metric without the dispatch version-stamp proving remote == local HEAD.
12. **Acceptance criteria must name the EXACT gated metric key (inherited):** copy it from the gate
    code; budget an independent adversarial verify for every gate-adjacent claim.
13. **Lane-isolation reality (inherited):** local lanes run file-fenced in the shared checkout;
    worktree-per-lane mandatory for VM/rsync contexts and any two lanes touching the same files.
14. **Wave-end docs reconciliation is MANDATORY (inherited):** every tennis wave closeout refreshes
    `CAPABILITIES.md`, ticks this file's checkboxes, and updates the PART VI wave log. A tennis plan
    that diverges from reality poisons every future session's boot.
15. **[TENNIS-SPECIFIC] Never regress pickleball.** Tennis is an ADDITION to a shipping product. Every
    tennis change to a shared file is gated on the full pickleball suite staying green. The
    sport-config seam is the contract: add a branch, never mutate the pickleball branch.

# PART V — EVIDENCE MAP

- **Parent plan (the engine being extended):** `NORTH_STAR_ROADMAP.md` (pickleball master),
  `TECH_BLUEPRINTS.md` (per-pillar executable specs), `EDGE_PLAYBOOK.md` (hacks + BOM),
  `CAPABILITIES.md` (canonical truth matrix), `TECH_STACK.md` (registry).
- **Tennis codebase audit (sport-genericity + delta surface):** `runs/lanes/tennis_recon_20260707/`
  — the file-level map of what already generalizes (`court_templates.py` tennis template,
  `Sport` literal, generic player roles) vs. what is hardcoded pickleball (ball physics, paddle
  rectangle, shot taxonomy, iOS), with the priority punch list.
- **Tennis research sweep (this doc's PART II):** `runs/research_tennis_20260707/` — 26 dimensions,
  ~60 Opus agents, adversarially verified: `{ball_tracking, ball_physics_3d, bounce_events,
  court_geometry, court_detection, player_tracking, body_pose, serve_biomechanics, stroke_class,
  racket_6dof, spin, multi_surface, fusion, live_tier, rules_scoring, calibration_capture, datasets,
  broadcast_engine, coaching_content, grounded_llm, competitive, market_gtm, speed_cost, legal_ip,
  player_styling_3d, eval_gt}_report.md` + the 5 synthesis blocks (verdicts, delta-matrix,
  strategic, competitive, risks/eval).
- **Key external existence-proofs cited throughout:** TrackNet (tennis-origin ball tracker),
  WASB-SBDT (tennis in-domain), TennisCourtDetector, Vid2Player / Vid2Player3D (Stanford — monocular
  3D tennis reconstruction from broadcast), Hawk-Eye/ELC (officiating reference bar), SwingVision
  (on-device iPhone incumbent). Full citations in the per-dimension reports.

# PART VI — TENNIS WAVE EXECUTION PLAYBOOK

Same wave lifecycle as the pickleball program (`NORTH_STAR_ROADMAP.md` §VI.0 — BOOT → PICK →
DIAGNOSE → PROVISION → DISPATCH → RULE → INTEGRATE → ADJUDICATE → FRESH-GPU PROOF → CLOSE), run
verbatim. Only the wave CONTENT differs. **Tennis waves ≤3 below are commitments; waves ≥4 are planned
trajectories** — each wave's closing scorecard re-derives the next wave's exact queue from measured
tennis evidence (the milestone mapping TM1–TM5 is the stable part, not the lane lists). PHASE TL
(live tier) rides alongside as an advisory stream and never delays the DATA→BALL→SERVE→coaching
critical path.

## VI.0 The invariant — read §VI.0 of the pickleball roadmap; it is unchanged.
Tennis adds one boot step: **assert the sport-config seam is wired** (T0 landed) before dispatching any
lane that computes tennis geometry — until T0 lands, a "tennis" run silently computes pickleball
constants (PART IV rule 8) and no result is trustworthy.

## VI.1 T-WAVE 1 — WIRE THE SPORT SEAM + FIRST HONEST TENNIS WORLD (TM1: "it runs on tennis")
The pickleball program spent wave 1 proving the machine ran; tennis wave 1 proves the machine runs
**as tennis, end to end, honestly banded** — even if every band is low-confidence. This is cheap and
decisive: it converts the half-built scaffolding into a real, gated tennis path.
- **T0-1 Sport-config seam** (finish the tennis branch: `court_corner_review` stub, tennis keypoint
  schema, residual-length validation) · **T0-2 wire `StageContext.sport` → `ball_arc_solver.court_sport`
  and all ~12 default sites** · **T0-3 tennis `PhysicsParameters` stub** (real constants land T1) ·
  **T0-6 fresh tennis E2E world** on one owner/broadcast clip with everything trust-banded.
- Gate (TM1): `process_video.py --sport tennis` produces a complete bundle on a tennis clip with the
  tennis court/net rendered, players placed, ball chain attempted, all bands honest; the full
  pickleball suite stays green. Kill: the seam can't be wired without a rewrite of a shared stage →
  escalate a scope decision, don't force it.

## VI.2 T-WAVE 2 — BALL + COURT TRANSFER (TM2: "the ball and court are basically there, cheaply")
The strategic bet: because tennis is the native domain of WASB/TrackNet and TennisCourtDetector, ball
2D and court calibration should reach bar with far less in-domain data than pickleball needed. This
wave tests that bet.
- **T1-1 ball transfer eval** (measure WASB-tennis + our zoo on owner tennis clips zero-shot; quantify
  the gap to bar) · **T1-2 tennis fine-tune only if needed** · **T4-2 tennis court detector**
  (retrain/adapt TennisCourtDetector head) · **T4-0 tennis court profiles** (owner's home courts).
- Gate (TM2): held-out tennis ball F1 + court PCK clear the tennis bars (I.2); the transfer bet is
  confirmed or refuted with numbers. Kill: transfer fails as badly as pickleball → fall back to the
  owner-capture data-engine loop, re-scope timeline.

## VI.3 T-WAVE 3 — THE SERVE PILLAR + RACKET 6-DOF (TM3: "it sees the serve")
Serve is tennis's signature. This wave stands up the new TS pillar + the racket-geometry rework.
- **TS-1 serve segmentation + toss track + contact detection** · **TS-2 serve kinematic-sequence
  features from 3D pose** · **TS-3 serve-speed estimate** (radar cross-check GT) · **T3-1 racket
  detection + strung-face model** · **T3-2 racket 6-DOF** (reuse PnP math, new silhouette/keypoints) ·
  **T3-7 sensor-racket GT session** (Babolat/Zepp/Sony as swing-speed/spin ground truth).
- Gate (TM3): a serve card — toss, contact height, racket-head speed, serve speed, foot-fault
  advisory — on an owner serve clip, every number gate-passed or trust-banded. Kill: racket 6-DOF
  can't beat the render-only bar → keep serve on pose-only signals, defer racket 6-DOF.

## VI.4 T-WAVE 4 — SPIN + MULTI-SURFACE + FUSION (TM4: "one consistent world, spin-aware")
- **T1-4 tennis 3D flight** (tennis Cd/Cl/Magnus) · **T1-5 spin estimate** (topspin/slice sign +
  magnitude band) · **T4-x multi-surface** (clay/grass/hard court detector + surface-specific bounce +
  clay ball-mark feature) · **TF-1/TF-2 fusion at tennis speed** (ball-meets-strung-face contact
  coupling).
- Gate (TM4): fused tennis world where ball meets racket at contact, spin sign agrees with bounce
  behavior, surface is classified and its physics applied.

## VI.5 T-WAVE 5+ — COACHING + PRODUCT (TM5: "it coaches my tennis, and a friend can use it")
- **T6 tennis shot taxonomy + reference-range library (NTRP/UTR bands) + grounded-LLM tennis coach
  (zero fabrication)** · **T7 tennis product mode** (iOS strategy per PART 0 sport-config ruling,
  pricing vs SwingVision, onboarding).
- Gate (TM5): a fabrication-audited tennis coaching card with ≥5 finding types tied to 3D moments,
  plus a friend onboarding into tennis mode.

## VI.6 Standing per-wave invariants (tennis)
Every tennis wave: (a) never regress pickleball (rule 15) — full parent suite green; (b) tag every
clip with a surface; (c) copy exact gated metric keys (rule 12); (d) independent adversarial verify on
every gate claim; (e) fresh-GPU proof with version-stamped code sync; (f) docs reconciliation +
`[T-WAVE-N COMPLETE]` scorecard + next boot prompt at close.
