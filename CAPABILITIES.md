# Capability Truth Matrix

Last updated: 2026-07-09.

This file records what the current repo can honestly claim. It is not a run log
and not a place for old experiment narratives. `BUILD_CHECKLIST.md` owns the
small operational board; `RUNBOOK.md` owns commands.

`VERIFIED=0`. Scoped passes remain scoped until the named gate passes.
CAPABILITIES.md is canonical on doc conflicts. The July 2026 GPU state is
reset-pending during winddown; do not treat any named A100 VM as currently
available without a fresh runtime check.

## Wave-7 Close Truth Sync

The 2026-07-09 wave-7 landings below are wired or ruled only within their
named scope. They do not soften `VERIFIED=0`: paddle and ghost meshes are
preview/ESTIMATED, ball checkpoint evidence is internal-val only, P2-2 remains
NOT-WIRING-READY, court auto-find ordering is unchanged, and security/licensing
reviews are pre-launch gates rather than product verification.

## Canonical Tier Split

This section is the single source of truth for live/server placement. `TIER_MAP.md`
is only a short mirror.

**L0 — LIVE IN-RALLY (on-device, during recording, <300ms).** Advisory
overlays/haptics rendered while the rally is being played. Cadence-scheduled ANE
inference on the live camera tap.

**L1 — LIVE BETWEEN-RALLY (on-device, seconds after a rally/recording stops).**
Instant replay of the last rally with overlays, challenge-style bounce zoom with
an uncertainty band, post-stop summaries. SwingVision's Review-mode pattern —
the proven single-camera UX.

**L2 — SERVER FAST VERDICT (~1-2 min after upload; to build).** The pipeline
WITHOUT the BODY stage (BODY = 97-98% of E2E wall). Ball chain + calibration +
placement + events + the line-call / court-call artifacts. Tighter than the
phone, still trust-banded, not promotion-grade.

**L3 — SERVER DEEP WORLD (authoritative, ~9 min today → 6-8 min booked).** The
full pipeline incl. SAM-3D BODY, global smoothing/association, arc refinement,
fusion, stats, coaching. The ONLY tier whose outputs can pass gates / be
promoted / earn VERIFIED. This tier is SAM-3D-Body only: RTMW/RTMW3D/RTMPose
are retired because optimized Fast SAM-3D-Body gives better pickleball-joint
accuracy at equal-or-better speed.

Naming continuity: existing docs' "ON-DEVICE LIVE / fast tier" = L0+L1;
"SERVER OFFLINE / deep tier" = L3. L2 is new, unlocked by the measured
stage-runtime map. Camera-space mesh preview is `server-fast`, not
phone-real-time. LiDAR is a near-field bonus only and is not required for v1.

**The live-tier constitution:**

1. **Live calls are ADVISORY, always.** Every L0/L1 call carries an uncertainty band and defaults to
   `too_close_to_call` when the margin does not clear sigma (the same fail-closed semantics as
   `decide_court_boundary` / `ball_inout_uncertainty`). Industry precedent: In/Out's "Too Close To
   Call" alert; SwingVision's explicit no-call fallback. Never market live calls as officiating.
2. **The record path is sacred.** Live features run on a fail-open tap (`LiveFrameTap` attach()
   returning false = silent no-op) and back off cadence under thermal pressure; recording never
   degrades. Evidence: SwingVision's #1 churn driver is recording freezes (marlvel intel 2026-05).
3. **One decision layer, tier-agnostic.** The SAME rule/geometry code (`decide_court_boundary`,
   `ball_line_calls`, `shot_rules`, two-bounce/`excess_bounce`) serves every tier; tiers differ only
   in perception quality → sigma width → how often calls resolve vs abstain.
4. **Live never trains, gates, or promotes.** VERIFIED semantics unchanged; L0/L1/L2 outputs never
   feed detector metrics, gates, training, or promotion.
5. **Device budget law.** FP16 + `.cpuAndNeuralEngine` (2-4x win, corroborated); cadence-scheduled
   inference (1-in-N frames + tracker/Kalman between); person 640px, ball 288-512px; plan with the
   measured 4-12x sustained-vs-burst live-camera tax; NOTHING runs per-frame at 240fps (4.2ms
   budget) — record high-fps, infer at ≤60Hz cadence. An in-house record+infer soak benchmark is
   mandatory before any latency promise (confirmed literature gap — no published number exists).

**Per-feature tier matrix:**

| Feature | L0 in-rally | L1 between-rally | L2 fast verdict | L3 deep world | How + unlock |
|---|---|---|---|---|---|
| Capture guidance (framing/level/exposure/fps) | TODAY (computed; UI unwired) | TODAY (post-stop summary computed; UI unwired) | — | — | `LiveGuidanceEvaluator` + `PostStopPreviewSummary` exist; wire to Record screen (PL-3) |
| Player detect/track + foot rings | TODAY (YOLO26n INT8 640, 1-in-4 cadence, tested live during recording) | replay overlay | — | authoritative tracks (BoT-SORT + global re-assoc) | bundle model in app (PL-3); screen-space today |
| Live court geometry (homography) | NEXT (tap-corner seed pre-record; ARKit plane assist) | same | manual taps / metric-15pt / profiles | metric-15pt + distortion | THE keystone: no line-relative call at any tier without it (PL-1; H0 court profiles make it one-time per court) |
| Court-plane dot map (real, not proxy) | after PL-1 | after PL-1 | TODAY (placement) | TODAY (RTS-smoothed) | screen-space proxy today, honest note in code |
| Kitchen-proximity indicator (feet vs NVZ line) | after PL-1 (geometry only; explicitly NOT a fault verdict — volley state unknown live) | after PL-1 | after PL-6 wiring | full NVZ fault w/ volley-state + momentum review-flag | `decide_court_boundary(near_kitchen)` exists unit-tested; NVZ momentum (Rule 11.A.2) has no fixed time window → even L3 ships it as a review FLAG, not a verdict |
| Serve foot-fault advisory (Rules 7.A.1-3) | R&D after PL-1 + serve-moment cue (audio/pose); wide sigma; behind-court occlusion honest | replay w/ freeze-frame at serve contact | after PL-6 (contact event + foot pos) | contact-frame foot-vs-baseline call, trust-banded | pro reference: Hawk-Eye dedicates 6 close cams to foot faults; single far phone = advisory ceiling |
| Ball trail + rally segmentation | after ball student trained (PL-5; blocked on P1 bar + owner data) | after PL-5 | TODAY (WASB, near-streamable 3-frame windows) | TODAY | student deploy path proven (1.41ms ANE); model untrained + kill-switched today |
| Ball in/out advisory | LATER: coarse bounce-zone dot (needs PL-1 + PL-5); expect 9-30cm-class monocular bounce error → most near-line balls = too_close_to_call | challenge replay + advisory call + uncertainty band (the flagship L1 feature) | `ball_line_calls` + `ball_inout_uncertainty` wired into orchestrator (PL-6) | arc-solver-refined call + 75%-rule-style framing | reference points: Hawk-Eye 2.2-3.6mm/10-12 cams; SwingVision 97% within-10cm band claim at 60fps; ball-on-line=IN except kitchen line on serve (Rule 8 + 7) |
| Two-bounce / double-bounce (Rule 10) | LATER (needs live bounce events; PL-5) | after PL-5 | `excess_bounce` in shot_taxonomy (wire via PL-6) | TODAY (lib) — wire | cheap state machine once bounce events exist |
| Score / serve-side tracking | manual/voice entry v0 (UI, no CV) | manual + inferred assist | — | H26 inference after P6-1 shot taxonomy (PL-7); later distilled down | does not exist anywhere today (honest) |
| Ball speed | — | rough 2D+court-scale estimate after PL-1/PL-5 | from solved arcs | authoritative from 3D arc | fps math: 60fps sufficient for mph-class estimates (baseball precedent 3.6mph err) |
| Shot types (9-rule table) | — | — | after PL-6 (needs S1 features) | P6-1 chain | `shot_rules_v0.json` landed, unwired |
| Highlights / instant replay clips | rally-end trigger (motion+audio heuristic) | TODAY-ish: replay last capture; contact-density selection later | — | authoritative highlight reel | rally_gating logic is cheap OR-fusion, portable |
| Stats + coaching card | — | — | rally counts/basic stats | P6 coaching (grounded-LLM, fabrication-audited) | offline-only by design |
| 3D world / mesh replay | NEVER live (BODY = 97-98% of wall, A100-class) | — | — | TODAY | the permanent offline anchor — also our accuracy moat (EDGE "we are offline" advantage stays true for L3) |
| NVZ momentum fault (11.A.2), partner-contact (11.A.1) | never (judgment-call rule) | — | — | review-flag only | rulebook: momentum ends when "balance and control" regained — no fixed window; flag, human decides |

**Dependency spine for the live tier:** PL-1 live court lock unlocks every
line-relative feature (kitchen proximity, serve position, bounce zones, real
dot-map). PL-5 trained ball student, gated on P1 hitting the internal bar and
owner in-domain data, unlocks every ball-relative live feature. PL-2 soak
benchmark gates every latency/cadence promise. These run parallel to the BALL
critical path and must never delay training waves.

## Capability Matrix

| stage | named tech (registry) | actually invoked? | correct variant+weight? | wired into spine? | gate type (accuracy/presence/none) | gate run on real labels? | honest status |
|---|---|---|---|---|---|---|---|
| calibration | manual/metric sidecar, ARKit target, OpenCV solvePnP, court detector candidates | manual/metric paths can feed `process_video.py`; wave-7 court owner exports corrected the GT pin to `gt_corpus_manifest_r2.json` md5 20aa6da46315888265a9487626c9224b and added exactly one net-new usable full-15kp frame, but all 4 scoreable checkpoints still scored 0.0 PCK@5 on the changed source (`runs/lanes/w7_courtkpingest_20260709/report_r2.json`; BUILD_CHECKLIST `[W7 COURTKP R2 (task-13 owner ruling executed) 2026-07-09]`) | metric sidecar is data, not a model; candidates remain unpromoted; court_unet_v2/E4 remain PENDING in `best_stack.json` until their 0.95 PCK gate | yes for sidecar/metric; preview detector can seed taps only; `run_ball_chain --court-calibration` is the handoff seam (83e090168; `runs/lanes/w4_court_harvestcal_20260707/report.json`) | held-out court PCK/reprojection | yes for prior owner-gate attempts; failed; corrected wave-7 labels do not change the court ordering or the P4-0 profiles-first ruling (`runs/lanes/w7_courtkpingest_20260709/report_r2.json`) | SCAFFOLD/PREVIEW, not VERIFIED |
| tracking | YOLO26m, BoT-SORT/ReID, OSNet, raw-pool association | runner/tooling exist; pre-registered gate runs still fail required coverage/identity/spectator constraints | manifest-checked where runtime uses weights | yes, with explicit reuse and association modes | IDF1, ID switches, spectator/background FP, off-court FP, coverage | yes; no candidate promoted | IN-PROGRESS, not VERIFIED |
| ball | WASB, TrackNetV3 family, audio onset, bounce/in-out, event fusion, default 3D arc chain | runtime/tooling and reviewed-label utilities exist; wave-7 ingested the second owner ball export, moving reviewed rows 1,121 -> 1,750 with all skips accounted and visibility-level ruled uninformative for w6-session rows (`runs/lanes/w7_ballingest2_20260709/report.json`; BUILD_CHECKLIST `[W7 BALLINGEST2 PASS 2026-07-09]`, `[W7 OWNER RULINGS INGESTED (label-session notes) 2026-07-09]`) | default remains the raw WASB tennis anchor sha256 9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb; `A_seed_official_aug` is only a PENDING candidate with md5 cfda3c423e1f93c0db42f20e32bdae9e and named held-out+owner gate in `configs/racketsport/best_stack.json` rev9 | yes, with reuse, confidence-banded display, fail-closed arc sanity, BVP v2, and full-rate `ball.detection_stride=1`; scalar Magnus/spin remains DORMANT (`runs/lanes/w6_magnus_20260708/verify/reprojection_compare.json`; `runs/lanes/w7_cadence_20260709/report.json`) | reviewed ball F1, bounce/contact timing, in/out agreement | partial reviewed labels exist, but the 1k-checkpoint card is internal-val only: A 0.6152 F1 / 0.2506 hFP, C 0.6121/0.2597, control 0.3611/0.5991; label curve 486->1121 is +15.5% relative F1 and not plateaued; D no-aug shows occlusion aug mitigates hFP, not causes it (`runs/lanes/w7_ballretrain_20260709/REPORT.md`; `runs/lanes/w7_ballscore_verify_20260709/report.json`; `runs/lanes/w7_ballcomplete_20260709/REPORT.md`) | SCAFFOLD, not VERIFIED |
| body | Fast SAM-3D-Body, body-frame scheduler, grounding refinement, mesh index export | remote/local BODY paths, mesh index artifacts, speed improvements, visual smoothing, and camera-motion AUTO decode-orientation telemetry exist for scoped clips; P2-2 production now applies `pred_cam_t` exactly once, the canonical harness now passes/fails closed on missing `pred_cam_t`, and synthetic-gate truth authoring was repaired, but the decisive GPU measurement still leaves the canonical GATE-1b 262.348mm/53.50mm p95 class unresolved; raw `pred_keypoints_3d` with cam_t matches persisted joints_world at p95 23.4mm/max 26.7mm, so this is a measurement/ceiling problem, not a wiring promotion (`runs/lanes/w7_p22checklist_20260709/report_r3.json`; `runs/lanes/w7_p22gate_20260709/gate1b_raw_arm_report.json`; `runs/lanes/w7_gatecheckfix2_20260709/report_r2.json`; BUILD_CHECKLIST `[W7 P22 GPU MEASUREMENT DECISIVE 2026-07-09]`) | runtime configuration is explicit; additive `scale_params` schema threading exists; candidate-label review is not GT; P2-4 SAM-Body4D was NO-ATTEMPT due permission/HF access, with only a 307.179s H100 baseline banked (`runs/lanes/w7_masklet_20260709/REPORT.md`) | yes, writes BODY/world contracts when inputs exist; BODY dispatch requires explicit remote host and version stamps; base `body.skeleton_stride=2` is wired for 30Hz effective cadence with contact-dense boosts preserved (`runs/lanes/w7_cadence_20260709/report.json`); latent smoother, grounding_refine un-kill, and latent-interp playback remain blocked | independent-GT world-MPJPE plus decode-fidelity synthetic round-trip for P2-2 | external/candidate paths have not produced a promotion; visual metrics, H100 BODY, stride-2 cadence, and P2-2 harness fixes are scoped/runtime evidence only; P2-2 remains NOT-WIRING-READY and `VERIFIED=0` unchanged | SCAFFOLD, not VERIFIED |
| foot/physics | foot-lock, render-honest ball fill, placement chain | scoped internal-val chain exists; skeleton-direct per-foot phase producer landed UNWIRED and measured negative for gate prediction: 3/4 clips breach the frozen 30mm max gate, so raw skeleton noise remains the binding constraint and grounding_refine stays an honest no-op (75e438223; `runs/lanes/w4_footattr_fix_20260707/report_r2.json`) | deterministic artifacts, no flagship learned dynamics lock; forbidden gate-referencing phase exclusion was removed (75e438223; `runs/lanes/w4_footattr_fix_20260707/report_r2.json`) | partial, consumed into world artifacts when present; process_video wiring remains fenced/unapplied for the new producer (75e438223; `runs/lanes/w4_footattr_fix_20260707/report_r2.json`) | slide/penetration plus protected replay validation | internal-val only | INTERNAL-VAL DONE, not VERIFIED |
| racket | detector/segmenter candidates, PnP-IPPE, fused wrist/palm paddle estimator, future reference-pose stack | phase-1 fused estimator is now default-wired through `process_video.py` when SAM-3D wrist/palm evidence exists and fail-closes with no artifact plus a structured `paddle_pose` summary when evidence is absent (`runs/lanes/paddlewire_p31_20260709/report.json`; BUILD_CHECKLIST `[W7 P3-1 PADDLEWIRE PASS 2026-07-09]`) | `paddle.fused_estimator` is `WIRED_DEFAULT` in `best_stack.json`; artifact remains `racket_pose_estimate.json` with source `wrist_palm_grip_fused` and trust band `estimated_preview`; no reference-pose GT model is approved | yes for the ESTIMATED render-only path; `--no-paddle-pose` opt-out is wired; reflection cone factor stays DORMANT until P1-4 real 3D ball velocities (`runs/lanes/paddlewire_p31_20260709/report.json`) | face-angle/contact error versus true reference GT | no true-corner/reference GT yet; internal-val parity only (Wolverine 0.224126 vs banked 0.235558, Burlington 0.331265 vs 0.342387) | SCAFFOLD, not VERIFIED |
| metrics | biomech primitives, confidence calibration, report checks, BODY+COURT match_stats v0 | BODY+COURT-only match stats v0 landed as a post-hoc consumer: per-player distance, speed p50/p95, heatmap grid, kitchen/baseline/transition zone time, and left-right balance, with ball/paddle-derived stats refused until those inputs pass (`runs/lanes/w7_p62stats_20260709/report.json`; BUILD_CHECKLIST `[W7 P6-2 STATS V0 PASS 2026-07-09]`) | no user-facing authority model; stats are trust-banded from source artifacts and exclude ball/paddle/rally claims | wired default-on/fail-open through `stats.match_stats_v0` in `best_stack.json` rev7 and process_video integration (`runs/lanes/w7_pipepolish_20260709/report.json`) | reviewed metric correctness | internal-val only on Wolverine/Burlington banked runs; no product/user verified stat gate | SCAFFOLD, not VERIFIED |
| shot/drill | pose/ball event features and classifier candidates | scaffold/eval utilities exist | no pickleball-approved model | not an authority path | reviewed shot taxonomy accuracy | no | SCAFFOLD, not VERIFIED |
| replay | Three.js review viewer, GLB/USDZ/replay manifest, mesh index viewer path, trust bands | scoped web/native artifact paths load review bundles; wave-7 completed ghost meshes end-to-end: human_review/manual_review_required frames can enter mesh eligibility, count against the byte budget, and emit `trust_badge=preview` through frame_compute_plan -> BODY -> mesh export -> body_mesh_index (`runs/lanes/w7_ghostviewer_20260709/report.json`; `runs/lanes/w7_tierprov_20260709/report_r2.json`; BUILD_CHECKLIST `[W7 TIERPROV PASS 2026-07-09]`) | scoped artifacts only; wider tier-eligibility raise and 300-vs-400 display policy remain owner-ruling PENDING in `best_stack.json` | yes for review surface and preview ghost emission; production native path still gated | structural, visual, FPS, native/web playback | scoped visual/check suites only; ghost preview is not BODY quality verification | SCAFFOLD/SCOPED PASS, not VERIFIED |
| ios live tier | LiveFrameTap, LiveCourtOverlayEngine, CoreML/ANE person detector, ball heatmap spike, LiveGuidanceEvaluator, PostStopPreviewSummary | built spikes plus live person overlay wired; guidance/summary computed but not fully rendered; ball UI kill-switched | person model is not bundled; ball student is untrained/kill-switched; no live court homography | L0/L1 advisory only; upload priors/sidecar feed later L2/L3 work | P0-10 device smoke, PL-2 soak benchmark, and presence/latency gates only | no; P0-10 device smoke pending; live outputs never train/gate/promote | SCAFFOLD/SCOPED PASS, advisory only, not VERIFIED |
| e2e | `scripts/racketsport/process_video.py` plus manifest-owned best-stack defaults | can write complete/partial scoped bundles with fail-closed summaries; wave-7 added first-class input-quality preflight (`input_quality.json` + `PIPELINE_SUMMARY` block), default-on fail-open match_stats, paddle default wiring, ghost mesh emission, and manifest-owned cadence (`runs/lanes/w7_pipepolish_20260709/report.json`; `runs/lanes/w7_cadence_20260709/report.json`; `runs/lanes/paddlewire_p31_20260709/report.json`; `runs/lanes/w7_tierprov_20260709/report_r2.json`) | no full-stack approved variant set; best_stack rev9 now accounts for every wave-7 gain as WIRED_DEFAULT, PENDING with a gate, DORMANT/FENCED, or no-delta | yes, as the glue entrypoint; strict input quality can fail-close before heavy stages, advisory default emits `degraded_input`; body stride-2 is wiring, not an accuracy claim | artifact completeness plus every component quality gate | no full clean gate-passing clip yet; pipepolish/cadence full censuses had sandbox loopback-bind failures in-lane and no component quality promotion | SCAFFOLD/SCOPED PASS, not VERIFIED |

## Security And Licensing Gates

Wave-7 security review is a pre-launch gate, not a verified product claim: the
review found three HIGH launch blockers and requires networked dependency/secret
history scans plus staging auth/delete-cascade smoke before flags flip
(`runs/lanes/w7_securityreview_20260709/report.json`; BUILD_CHECKLIST
`[W7 SECURITYREVIEW P7-4c DONE 2026-07-09]`). Wave-7 licensing review rules
private/internal use as acceptable for current development, but monetization
must exclude/rebuild NC/unknown-trained checkpoints, resolve GPL/AGPL/SMPL and
platform/model-card blockers, and produce a commercial-clean source ledger
(`runs/lanes/w7_licensecheck_20260709/report.json`; BUILD_CHECKLIST
`[W7 LICENSECHECK P7-4d DONE 2026-07-09]`).

## Current Spine Reality

`scripts/racketsport/process_video.py` chains ingest, calibration, input-quality
preflight, tracking, placement, rally gating, frames, ball, the default ball-arc
stage, events, ball fill, BODY, placement refinement, grounding refinement,
paddle_pose, world, confidence gating, match_stats, manifest generation, and
optional viewer verification. It reuses valid artifacts unless `--force` is
set, degrades missing optional stages into trust-banded gaps, and writes
`PIPELINE_SUMMARY.json` for both complete and partial runs.

BODY dispatch is remote by default unless `--body-local` or `--no-gpu` is used,
but the winddown state is reset-pending and any remote host must be rechecked
before use. Current BODY outputs fetch `body_mesh_index/` for replay by default;
large `smpl_motion.json` and `body_mesh.json` monoliths are opt-in with
`--fetch-body-monoliths`.

The important honesty boundary is that `complete` means "a bundle was produced,"
not "the underlying CV has passed product gates."

## Protected Data Policy

- Outdoor and Indoor are strict held-out clips.
- Burlington and Wolverine are internal-val only when explicitly opted in.
- Roboflow/public corpora may be training data only after provenance/dedup checks.
- Candidate predictions, copied labels, and box-derived paddle candidates are not
  independent ground truth.

## Promotion Vocabulary

| Word | Meaning |
|---|---|
| `VERIFIED` | The documented acceptance gate passed on required current evidence. No row currently qualifies. |
| `SCOPED PASS` | A specific command/artifact/browser/device slice passed. It must name its scope. |
| `INTERNAL-VAL DONE` | A useful internal-val result. It does not promote held-out or product claims. |
| `SCAFFOLD` | Code/contracts/tests exist, but quality gates or runtime proof are missing. |
| `PREVIEW` | Usable for visual inspection or UX only; not authoritative. |
