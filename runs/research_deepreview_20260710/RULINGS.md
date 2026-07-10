# Deep review 2026-07-10 — owner symptoms S1–S5, dual-track (Fable × Codex gpt-5.6-sol)

Owner directive (2026-07-10): deep research on everything built so far, find bugs / things done
wrong, define next steps to the final goal; Fable and Codex each investigate independently with
their own agents and reconcile.

Method: 3 Codex gpt-5.6-sol lanes (`runs/lanes/dr_viewer_20260710/`, `dr_pipeline_20260710/`,
`dr_sota_20260710/`) + a 45-agent Fable workflow (7 sonnet finders → merge 52→47 defects →
2-lens adversarial verify on 16 critical/high → 9 CONFIRMED / 3 contested-refined / 4 REFUTED →
completeness critic → 4 gap probes, incl. frame inspection of the actual demo mp4 and an iOS
probe). Manager personally re-verified the 5 load-bearing wiring claims at HEAD (cited below).
Protected Outdoor/Indoor labels untouched. `VERIFIED=0` unchanged; nothing here is a promotion.

## S1 "frame rate still seems low" — three stacked causes; the biggest is the demo itself

1. **DECISIVE: the demo video was assembled at 10 unique frames/s.** World segments were built
   from headless screenshots at 0.1 s steps (`demo_beststack_render_20260710/concat_after.txt`,
   `world_after_cleancrop.log`: 10 fps / 102 frames) then upconverted into a 30 fps container.
   What the owner watched is not the product's playback.
2. **Every headless "3D FPS" number in the repo (1.2–3.5 fps) is SwiftShader software-WebGL
   dominated** (zero-player world reads 46 fps in the same harness). These numbers must never be
   quoted as app performance again. **No real-device FPS measurement exists anywhere** — gap.
3. **Real cadence stepping exists regardless of GPU:** every entity does nearest-frame lookup with
   holds (mesh hold 0.15 s, edge fade 0.12 s, no interpolation between mesh keyframes; the 2x-FPS
   toggle is off by default and only inserts midpoints). Wolverine carries 244 unique mesh times
   over 10 s → visibly steppy at any renderer speed.

Fix chain: real-device profile first (owner iPhone ask) → display-time interpolated playback
(video PTS master clock; rVFC + rAF; slerp/lerp between bracketing samples; refuse interpolation
across trust/contact/entity discontinuities) → mesh chunk prefetch (today: 2-entry LRU, no
prefetch, reactive fetch ~0.4 s cadence) → demo tooling must record real-time canvas.

## S2 "some people aren't detected" — the detector is mostly fine; three droppers sit on top

Coverage waterfall (dr_pipeline, per-frame counts on real artifacts): raw YOLO/BoT-SORT sees ≥4
people on **96.7%** of Wolverine frames (78.1% owner-clip) → zero-margin court filter leaves
**25.7%** (5.4%) → top-4 stable-ID selection leaves **13%** (**0%** — the owner clip's four
selected IDs never coexist in any frame).

1. **Zero-margin court-membership filter** (`threed/racketsport/person_fast.py:43-49`,
   `scripts/racketsport/track.py:93` default `court_margin_m=0.0`; manager-verified). Feet
   projected 1 cm behind the baseline ⇒ player deleted. Pickleball players *live* behind the
   baseline; this is structural, not tuning.
2. **Best-stack ReID/OSNet asset missing ⇒ silent degradation**: `_attempt_global_association`
   quietly "kept loose-pool tracks.json" when `--reid-model` is absent
   (`scripts/racketsport/process_video.py:1589-1597`; manager-verified verbatim). The intended
   association path never ran on the audited clips. Silent best-stack degradation violates
   fail-loud (standing rule 10).
3. **Top-4-by-stability selection** drops fragmented real players (owner clip: raw IDs 15/45 keep
   337/266 on-court frames but are unselected).
4. Viewer (secondary): world frames with all-null pose render **nothing** — no placeholder, no
   "lost track" state (player 20: 93 consecutive null-pose frames = 3.1 s; 3 of 4 players have
   40+-frame mid-clip null runs in the demo world). Solid-mesh tier is separately sourced from the
   mesh index (adversarial-verify refinement) and needs its own no-coverage handling.

## S3 "should ALWAYS be skeleton or mesh" — the no-fallback hypothesis is REFUTED

The viewer **already has** the fallback ladder: BODY joints → real skeleton; else floor/track
anchor → proxy skeleton (`web/replay/src/App.tsx:2139-2143`, `3079-3117`; manager-verified).
It is starved by data:

- Owner clip: **zero** skeleton/mesh because BODY never ran — the cold-clip frame-materialization
  bug, **fixed the same morning** by ns016 (`7a6fd828e`); the demo predates the fix.
- Residuals: (a) the 1,200-frame materialization cap still excludes 115/1,315 tracked owner-clip
  frames — violates the always-representation rule; needs an explicit cap policy ruling (NS-04.2);
  (b) manifest labels total-BODY-absence as `skeleton_only` (`process_video.py:3819-3835` checks
  only mesh artifacts) — honest-status lie, NS-01.5; (c) Wolverine still has 19.7–49.7% per-player
  frames with NEITHER geometry — that is S2's TRK gaps flowing downstream; (d) two small viewer
  bugs: all-implausible moments switch the whole skeleton layer off, and `skeleton_implausible`
  suppresses the proxy too (`App.tsx:2143`; manager-verified).

## S4 "ball is hidden SO much" — fail-closed works; the waste is the 2D→3D lift + dead wiring

- **71–91% of our visible 2D ball detections never survive to emitted 3D** (Wolverine 243 visible
  2D → 75 emitted 3D; owner clip 480 → 58). pb.vision emits 69% by *omission* (0.58% flagged
  interpolated); our 2D coverage 80.6% beats their 58.7% — the headroom is ours to lose.
- **Built-but-dead wiring found:** segment-bridge interpolation (`physics_interpolated`) exists but
  `rally_gating: bool = False` keeps it dead (`process_video.py:454`; manager-verified); the
  RANSAC inlier gate module exists (`threed/racketsport/ball_ransac_arc_gate.py`) but has no
  reference in the default pipeline (manager-verified); BlurBall-style cue similar. The
  UKF-seeded fallback (adopt #5) and TT3D joint-anchor search (adopt #1, the centerpiece) are
  **unbuilt**.
- Viewer: predicted styling exists (dashed cyan/blue/amber + pulsing marker + HUD) but the
  `physics_predicted` provenance band collapses to generic styling; `Ball`/`BallGhostMarkerRing`
  are dead code (never instantiated — adversarial-verify catch); the trail builder can bridge
  suppressed spans (ignores segment IDs up to 1.75 s) — high-risk defect; current marker requires
  a sample within 0.12 s.
- UKF ceiling math (counterfactual bounds, dr_pipeline): adjacency-gated UKF lifts Wolverine at
  most 75→234/300 but the owner clip only 58→186/1,350 — **anchor search is load-bearing, UKF
  alone is not enough on owner-class clips.**

## S5 "paddle looks really bad + mispositioned" — three confirmed layers

1. **Data:** 100% of paddle frames are `wrist_palm_grip_fused` render-only estimates — flat ~0.51
   confidence, 0 reprojection errors populated, no detector boxes, no contact locks; bypasses the
   typed `coordinates.py` API (P0-D). True accuracy fix = NS-03.RKT and needs gold-capture GT.
2. **Viewer staleness:** nearest-frame hold with NO internal gap bound (`viewerData.ts:2679-2687`);
   measured 1.467 s paddle gaps ⇒ up to ~0.73 s stale pose visibly detached from the swing.
3. **Presentation:** 16-vertex box proxy + a 0.52 m debug normal arrow rendered LONGER than the
   0.406 m paddle. Demo frame inspection confirms torso/pelvis-clipped paddles through ~20 s of
   the shipped demo. Coordinate-transform error remains an unproven hypothesis (declared frames
   internally consistent) — do not chase it before GT.

## S6 — additional confirmed findings the owner did not name

- **iOS replay screen is hardcoded to the bundled fixture** — it never loads real per-capture
  pipeline output (no manifest plumbing; iOS schema also lacks `ball_arc_render_url`). This is the
  concrete form of P0-B's "Open selects the local row". NS-01.2b/01.5.
- **BODY-failure review writer overwrites the authoritative `frame_compute_plan.json`**
  (`orchestrator.py:2756-2838`) — plan/summary disagreement observed on the owner clip. NS-01.6.
- **camera_motion carries parent-video frame indices into cut clips** (reference frame 109050 in a
  1,350-frame excerpt ⇒ stage always fails on harvest excerpts). NS-01.1/01.4.
- **Trust badges are unmounted** (`TrustBandPanel`/`PlayerTrustBandPanels` defined, never mounted;
  headless `trust_chip_count=0`) — the product's honesty mechanism is invisible; missing mesh badge
  defaults to solid material (untrusted looks trusted). NS-01.5/NS-05.4.
- One optional-artifact 404 aborts the entire viewer load; manifest URLs embed absolute machine
  paths; event/landing markers default off and share one active-count gate; per-entity seek
  tolerances disagree (seek-snap family).
- `unfingerprinted_stale` reuse loophole weakens content-addressing (P0-C). NS-01.3.
- Events-before-BODY confirmed live: `contact_windows.events=[]`, wrist peaks blocked, no
  post-BODY refinement pass runs (P0-G). NS-01.7 — the largest spine change.
- Both audited clips report `no_audio_stream` — excerpt cutting appears to strip audio; audio can
  contribute zero until NS-01.7 plumbing lands anyway, but ingest should preserve it.

## Refuted this review (do not re-litigate without new evidence)

1. "Viewer lacks a skeleton fallback" — REFUTED; it exists and is data-starved.
2. "Fail-closed ball renders with no distinct predicted marker" — REFUTED as stated; band styling
   exists (the real gaps: provenance collapse, 0.25 s horizon, dead `Ball` component).
3. "Cold-clip BODY bug still breaks owner clips at HEAD" — FIXED by ns016 same-day (cap residual
   remains, above).
4. "Paddle misposition = coordinate-transform bug" — unproven hypothesis; needs marker GT.
5. "Mesh policy hard-excludes <4-player frames" — mischaracterized (threshold/fallback differ).
6. "pb.vision ships a continuous 3D replay we must catch" — REFUTED (dr_sota, first-party): their
   public 3D is a static filterable shot chart + video overlays. A continuous honest world proxy
   would exceed every public competitor surface, not chase one.

## Ordered next steps (traced to North Star; nothing here weakens a gate)

| Wave | Scope (file-fenced) | Content | Gate/acceptance |
|---|---|---|---|
| V — viewer truth+quality (Codex, `web/replay/**`) — DISPATCHED | presence placeholder tier + null-pose handling; implausible-skeleton gate fixes; paddle gap-decay + normal-arrow off + cleaner proxy + preview badge; ball 4-state contract incl. explicit `physics_predicted`, segment-ID trail break, marker/count source unification; mount trust badges w/ fail-safe-to-preview; optional-asset isolation; event/landing marker defaults; unified PTS time-map for seek; chunk prefetch + cache bump | focused unit tests + headless decision-rule parity on the two audited worlds; NO fps/accuracy claim | NS-01.5/NS-05.4-adjacent presentation truth; no stack delta |
| P — pipeline honest wiring (Codex, `process_video.py` + `orchestrator.py` + manifest/status surfaces) — DISPATCHED | fail-loud (or explicit degraded provenance) on missing stack-pinned assets incl. ReID; `body_missing`/`track_only` manifest truth; side-effect-free review writers; camera_motion parent-frame remap or explicit-unavailable; persist fail-closed verdict map in a world-adjacent sidecar | red→green tests per behavior; wide suite MPLBACKEND=Agg; no promotion | NS-01.5/01.6 scoped; integration-owner slot (free) |
| T — TRK recovery scoring (GPU) | restore/ship OSNet asset + apron-margin candidates (e.g. 0.5/1/2 m) + association re-enable, scored ONLY by the frozen TRK scorer on the fresh-clip set | IDF1 ≥0.85 / 0 switches / coverage ≥0.95 family; no threshold shopping | NS-03.TRK step 1 |
| B — ball wiring + candidates (GPU) | flip `rally_gating` bridge, wire RANSAC gate + blur cue as CANDIDATES, build UKF-seeded fallback; all scored vs frozen ball gates with fail-closed intact; then build TT3D joint-anchor search (kill: fallback must drop <5/11 on the diagnosed clip) | F1@20 ≥0.90 / recall ≥0.75 / hFP ≤0.05 family + fail-closed suppression parity | NS-03.BALL + adopt-sequence §1-4 |
| M — measurement | Mac Safari + owner-iPhone live FPS trace of the Wolverine replay (hardware, not SwiftShader); demo tooling switched to real-time capture | honest device numbers before any renderer optimization | NS-06.1 slice |
| — | RKT accuracy, ball-3D GT, BODY world-MPJPE | blocked on gold capture (owner ask, standing) | NS-02.1/02.2 |

Priority ruling (reconciling Codex's "integration-first" ranking with the banked ball adopt
sequence): both are right — they own different symptoms. Waves V+P are presentation/honesty
truth work with no gate risk and the largest owner-visible delta; T and B are the accuracy levers
and run on GPUs behind frozen gates. The banked ball sequence is unchanged inside its lane.

## Owner asks (delta to the standing §5 table)

1. **10-minute iPhone FPS check:** open the staged Wolverine replay on your phone when we send the
   link (Wave M) — decides how much renderer work S1 actually needs.
2. **pb.vision source-clip identity** for the banked cv export (still open) — enables the
   exact-clip benchmark for every ball change.
3. Gold capture half-day (standing ask #2) — paddle accuracy and ball-3D truth are capped until it
   happens.

BEST-STACK DELTA: none (audit + rulings only). Fix lanes above flip nothing without named gates.

Evidence: `runs/lanes/dr_viewer_20260710/FINDINGS.md`, `runs/lanes/dr_pipeline_20260710/FINDINGS.md`
(+ coverage CSVs), `runs/lanes/dr_sota_20260710/RESEARCH.md`, Track A result JSON
(`trackA_result.json` in this dir), `runs/lanes/w7_pbv_compare_20260709/COMPARISON.md`.
