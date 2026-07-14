# Ball-Contact Event Label Bootstrap — Synthesis Report

Scope: public datasets/models with frame/time-level HIT and BOUNCE labels usable to train our pickleball event head (single fixed camera, 720p–4K @ 30–60fps, ±2-frame tolerance). Known names (WASB, TrackNet v2/v3, TOTNet, RacketVision, BlurBall, AdaSpot, MonoTrack, pb.vision) excluded per instruction.

Verification legend: **[CORROBORATED]** = survived the 2-vote adversarial refutation pass. **[UNCERTAIN]** = primary source was fetched by the finding agent but the claim did NOT go through the 2-vote refutation pass (only OpenTTGames, Extended OpenTT, P2ANet, and TTStroke-21 received refutation votes). Nothing here is marked VERIFIED.

---

## 1. Ranked adoption table (top 8)

| # | Dataset | Sport | Use for us | Event labels (granularity) | Label quality for ±2-frame spec | License: R&D / Commercial | Access | Status |
|---|---------|-------|-----------|---------------------------|--------------------------------|---------------------------|--------|--------|
| 1 | **jhong93/spot Tennis (E2E-Spot, ECCV 2022)** — https://github.com/jhong93/spot | Tennis (broadcast, US Open + Wimbledon) | **Primary pretrain** for the 2-class event head — only public racket-sport set with BOTH hit-contact and bounce at frame precision | 33,791 single-frame events, 6 classes: serve/swing ball-contact + ball bounce, near/far court; eval regime is mAP@δ=1–2 frames (matches our tolerance) | Excellent — exact taxonomy match (HIT + BOUNCE), 25/29.97fps matches our capture | Labels+code BSD-3-Clause (permissive, commercial-OK for labels); **videos self-fetched from 28 YouTube IDs** — broadcast pixels are copyrighted, trained-weight provenance needs NS-07.3 review | Labels live in-repo right now (data/tennis/{train,val,test}.json + videos.csv), no registration; verified live 2026-07-13 | [UNCERTAIN] (no refutation vote, but repo files fetched directly) |
| 2 | **OpenTTGames (OSAI)** — https://lab.osai.ai/ | Table tennis | **Primary BOUNCE pretrain** — fixed single side camera, closest capture geometry analog | 4,271 frame-exact events, 3 classes: bounce / net / empty. JSON frame→event. NO racket-hit class. +4/−12-frame context annotations (ball xy + seg masks) | Excellent for BOUNCE (frame-exact); 120fps → subsample to 30–60fps | CC BY-NC-SA 4.0 verbatim — R&D OK, **NC-flag for NS-07.3** | Direct HTTP, no registration; links probed alive (HTTP 200) 2026-07-13/14 | **[CORROBORATED]** (2 votes) |
| 3 | **Extended OpenTT Games (arXiv 2512.19327)** — https://github.com/moamal01/table_tennis_data | Table tennis | The HIT complement to #2: frame-keyed stroke JSON supplies racket-hit frames on the same videos → **HIT+BOUNCE from one fixed camera** | Frame-number-keyed JSON: 7 stroke techniques (serve/loop/block/push/flick/lob/smash) + lean + feet + rally outcomes, on top of base bounce/net events | Strong; one open check: whether stroke key = contact frame vs stroke onset (inspect JSON on download) | CC BY-NC-SA 4.0 verbatim — same NC flag | git clone (annotations in-repo) + download_videos.sh from lab.osai.ai; no registration; probed alive | **[CORROBORATED]** (2 votes) |
| 4 | **ShuttleSet + ShuttleSet22 (KDD 2023 / CoachAI)** — https://github.com/wywyWang/CoachAI-Projects | Badminton (BWF broadcast) | **Bulk HIT pretrain / scale**: ~36.5k + ~33.6k frame-exact hit events with hit xy, shot type, homography | Per-stroke frame_num verified frame-precise (raw set1.csv inspected, not 1s-quantized). NO bounce class (shuttle doesn't bounce) | Good for HIT; broadcast pans/cuts vs our fixed camera | Annotations MIT (repo LICENSE); videos NOT distributed — self-source BWF YouTube | Direct clone, no registration (ShuttleSet22 via CoachAI-Challenge Track 2 folder; direct path 404s) | [UNCERTAIN] |
| 5 | **F3Set-Tennis (CVPR 2025)** — https://github.com/F3Set/F3Set | Tennis (broadcast, 114 matches) | HIT-scale co-train: 42,846 frame-level ball-racket contact moments (5× jhong93), more surfaces/cameras | Frame-exact contact only, 8 attribute dims; **NO bounce labels** | Very good for HIT | **No LICENSE file** + paper says "strictly for academic research" — R&D OK under our policy, must replace before launch (NS-07.3) | Labels in repo, videos via YouTube links; no registration seen | [UNCERTAIN] |
| 6 | **PadelTracker100 (Zenodo)** — https://doi.org/10.5281/zenodo.17020011 | Padel | **Closest domain cousin** (small enclosed court, doubles, net play, single fixed elevated camera, 30fps): domain-adaptation bridge; shot frames → generic HIT | 40,135 frames with 6-class shot events + per-frame ball boxes + poses + homography. **Contact-frame vs interval semantics UNVERIFIED** — check labels.zip. No bounce class | Unknown until precision check | **CC BY 4.0 — the only commercially clean license in the list** (labels); underlying WPT video must be sourced separately, rights not covered | Open Zenodo download, no registration (labels.zip 69.7MB) | [UNCERTAIN] |
| 7 | **CoachAI Badminton Challenge 2023 Track 1** — https://sites.google.com/view/coachai-challenge-2023/tasks/track1 | Badminton | Only set shipping **videos + HitFrame labels together** (no YouTube re-sourcing); challenge scoring tolerance was exactly ±2 frames (arXiv 2308.12645) | Per-rally MP4 + CSV incl. exact HitFrame, hitter, ball type, landing xy. No bounce | Good for HIT if granted | Not stated; BWF competition footage — assume research-restricted; NS-07.3 flag | **Gated Google Form** (https://forms.gle/znfgo4Bvp3t9h8wk9); 2026 availability unknown — one owner-time submission | [UNCERTAIN] |
| 8 | **BFMD — Badminton Full-Match Dense (CVPRW 2026)** — https://arxiv.org/abs/2603.25533 | Badminton (incl. doubles) | If released: only badminton set with HIT **and** floor-contact (landing) frames — the closest badminton BOUNCE analog; doubles = occlusion-rich like pickleball | 16,751 frame-level hit events + shuttle-landing first-contact frames + net hits, 19 matches / 20.3h | Good on paper | Not stated; BWF-sourced | **Release UNCONFIRMED** — no repo URL found; email authors before planning | [UNCERTAIN] |

**Below the line (evaluated, do not adopt):** P2ANet **[CORROBORATED]** (139k segments but ±0.5s windows that annotators could widen — cannot supervise ±2 frames; Baidu+email+deprecated); TTStroke-21 **[CORROBORATED]** (stroke start/end intervals only, and refutation found the license WORSE than reported: 2022 conditions mandated data destruction by Jan 2023 + face obscuring — PII gate; skip); TenniSet (intervals, tiny, drive-rot risk); THETIS (no temporal labels, no ball); PingTactics (paywalled, unproven access); FineBadminton (hit frames are model-generated — label noise vs our bar); BadmintonDB (intervals, no license); all Roboflow/HF/Kaggle/Rochester pickleball items (zero event labels); hudsong.dev (unreleased; cheap author-email option, low expected yield).

---

## 2. Recommended bootstrap recipe

**Stage 0 — Acquire (this week, ~zero friction):**
- Clone jhong93/spot labels + fetch its 28 YouTube videos (BSD-3 labels; keep videos out of the repo).
- wget OpenTTGames (5 train videos + markup) and clone Extended OpenTT annotations (NC-flag both in the NS-07.3 ledger).
- Clone ShuttleSet/ShuttleSet22 CSVs; download PadelTracker100 labels.zip and **inspect shot-event frame semantics** (the one cheap unknown that could promote it to rank ~3).
- Owner-time asks (small, parallel): CoachAI Track 1 Google Form; email BFMD authors; optionally email hudsong.dev for pickleball bounce CSVs.

**Stage 1 — Multi-sport pretrain of the event head:**
Train a 2-class (+background) temporal spotting head (E2E-Spot/T-DEED-style architecture — the jhong93 repo ships reference code under BSD-3) on the union: jhong93 tennis (HIT+BOUNCE), OpenTTGames+Extended (BOUNCE+HIT, subsampled 120→30/60fps), F3Set + ShuttleSet (HIT-only; mask the BOUNCE loss on these). Condition on the streams we already own: WASB 2D ball track + wrist/skeleton channels, so the head learns trajectory-kink + swing signatures rather than sport-specific pixels. Sport-balanced sampling; ±2-frame tolerant loss (Gaussian label smoothing over ±2).

**Stage 2 — Audio weak supervision on OUR captures (the big lever):**
- Build the audio pseudo-labeler (onset/transient detector on paddle "pop" + bounce thump; pickleball's rigid paddle is acoustically loud and distinctive).
- **Validate the audio→frame offset methodology first on jhong93 tennis**: its YouTube videos have audio and its GT hit/bounce frames are frame-exact — this gives us a free precision measurement of audio pseudo-labels before touching pickleball.
- Run it over product captures; keep only high-confidence audio peaks that coincide with a WASB trajectory discontinuity (audio ∧ kinematics agreement = pseudo-label; disagreement = mined hard example for human review).

**Stage 3 — Pickleball fine-tune + gold set:**
Fine-tune the pretrained head on audio-kinematic pseudo-labels; concurrently have the owner label a small gold eval set (~500–1000 events) in the existing CVAT flywheel — frame-clicking hits/bounces with ball track overlaid is fast. Gold set is for GATES, not training, until it grows. Optionally insert PadelTracker100 as a domain-adaptation step between Stage 1 and 3 if its contact semantics check out.

**Stage 4 — License hygiene (NS-07.3):** Commercial path keeps only: our own labels + audio pseudo-labels, PadelTracker100 (CC BY), jhong93/ShuttleSet/TenniSet label files (BSD-3/MIT — broadcast-video-trained *weights* still need review). NC (OpenTTGames family) and research-only (F3Set) contributions must be retrained-out or cleared before launch. Track all of it in the existing flag ledger.

---

## 3. What does NOT exist publicly (honest gaps)

1. **No public pickleball event dataset of any kind.** Zero hit/bounce/contact labels for pickleball anywhere (Roboflow, Kaggle, HF, academic — all checked and closed out). Our own captures + audio are the only path to in-domain labels.
2. **No dataset pairs audio with event labels.** Every source is video-only or ships labels without media; the audio-supervision lever has no public precedent to reuse — jhong93's YouTube audio is the closest thing to a validation bed.
3. **Bounce labels are rare, period.** Frame-level bounces exist only in OpenTTGames (table tennis) and jhong93 tennis; badminton has none (BFMD's landings, unreleased, are the only analog). HIT labels are plentiful; BOUNCE is the scarce class.
4. **No phone-camera / amateur-capture domain.** Everything usable is broadcast or industrial fixed cameras; the color/zoom/rolling-shutter gap to phone captures is unaddressed publicly.
5. **No videos+labels bundles with clean rights.** Every large set requires self-sourcing YouTube video whose pixels are separately copyrighted; the only fully-open-licensed labels (PadelTracker100) ship without video.

---

## 4. Verification markings (explicit)

- **[CORROBORATED]** (survived both refutation votes): **OpenTTGames** (with refinements: masks encode humans/tables/scoreboards not ball; "9-frame window" is a TTNet model detail); **Extended OpenTT Games** (refinements: +smash technique, fuller outcome vocab, video availability depends on osai.ai bucket); **P2ANet** (corrections: "loop" not a class; windows were annotator guidance and could be widened — even coarser than reported; adopted-against on merit); **TTStroke-21** (correction: license materially worse than first reported — campaign-scoped retention/destruction clause + PII; adopted-against).
- **[UNCERTAIN]** (primary sources fetched by the finding pass, but NO 2-vote refutation ran): jhong93/spot tennis, F3Set, ShuttleSet/ShuttleSet22, CoachAI Track 1, BadmintonDB, BFMD, FineBadminton, PadelTracker100, TenniSet, THETIS, PingTactics, and all pickleball-specific items. Highest-leverage residual unknowns: (a) jhong93 claims not adversarially checked despite rank-1 status — worth one refutation pass before committing GPU time; (b) Extended OpenTT stroke-key = contact-frame semantics; (c) PadelTracker100 shot-event precision; (d) Track 1 form and BFMD release viability; (e) all Roboflow licenses (pages 403 to fetchers — need browser check).
- **Explicitly speculative in inputs, repeated here as speculation:** P2A "MIT" covering videos; "800 training videos" figure for Track 1; PingTactics contents beyond its abstract; BadmintonDB ms-alignment (format inference).