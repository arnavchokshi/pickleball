# Events + Ball 2D→3D: the exact next steps, costed and gated

Lane: `next_steps_events_ball3d_20260728` (docs-only). Date: 2026-07-28.
`VERIFIED=0` remains binding everywhere in this document. Nothing here is a
capability promotion; it is an ordered execution plan grounded in the repo's
own measured evidence as of tonight.

**Scope note on freshness.** `NORTH_STAR_ROADMAP.md` was last updated
2026-07-26 and its active queue (§5) still reads as if rows 1, 3, 4 are
open. They are not — `main` HEAD (`6219514`) already contains a full
2026-07-27 work day that closed rows 1–4 at scoped-pass/engineering level
and made real progress on row 5. This plan reflects the state actually on
disk tonight, cites the lane evidence for every claim, and does not touch
`NORTH_STAR_ROADMAP.md` itself (out of fence for this lane). The single
highest-leverage unblocked action — E-v2 dispatch, queue row 6 — has not
moved, which is why it leads the "tomorrow morning" section.

---

## 0. Tomorrow morning — 5 concrete actions

Ordered by leverage. Costs are $/GPU-hour bands from measured or comparable
lane spend, not invoices.

| # | Action | Owner | Command / evidence | Cost |
|---|---|---|---|---|
| 1 | **Dispatch E-v2 scale-up + fine-tune.** The Step-0 training-data gate passes right now (verified live tonight, §4). This is the single highest-leverage unblocked action in the whole program. | agent (GPU dispatch), owner approves spend | `runs/lanes/trackD_ev2_design_20260722/VM_RUN_PLAN.md`, fixed at `f29145a`. Cut a fresh `RUN_COMMIT`, re-run Step-0 on that commit, then dispatch. | $2.2–4.5, 2–5h (A100), per `SCALE_UP_SPEC.md` |
| 2 | **Label 20–30 more bounces on the two GPU-free ready clips.** `outdoor_webcam_20s` and `burlington` are prepped and waiting; no GPU, no VM. | owner (human clicking) | `.venv/bin/python scripts/racketsport/ball_label_studio.py --run-dir runs/lanes/label_clip_prep_20260727/... --out runs/lanes/ball_label_tool_20260726/labels/<clip>` — see `runs/lanes/label_clip_prep_20260727/LANE_REPORT.md` §3 for exact paths | $0, ~1–2h |
| 3 | **`gcloud auth login`** — the external blocker named in queue row 7 (court/skeleton closeout) and needed for any fresh GPU dispatch tonight, including action #1. | owner (human, one command) | `gcloud auth login` (hello@) | $0, 1 min |
| 4 | **Re-run `verify_training_inputs.py` at the fresh RUN_COMMIT cut for action #1**, so the dispatch's Step-0 proof is not stale (proofs expire in 900s by design; tonight's proof is already past its 4-hour window by morning). | agent | `.venv/bin/python scripts/racketsport/verify_training_inputs.py --inputs runs/ball_lane_20260723/ev2_unblock/training_inputs_ev2.json --ledger runs/manager/data_ledger.json --repo-root . --gate-proof <fresh path>` | $0 |
| 5 | **Read the two cross-track handoffs waiting in `runs/manager/inflight_lanes.md`** before touching calibration or court code: the net-keypoint height defect (2026-07-27, affects any COURT retrain) and the shared-manifest write-lock convention in `.claude/skills/run-lane/SKILL.md` (calibpromo lane flagged a collision risk on `best_stack.json` revision). | agent | read-only | $0 |

---

## 1. Where the program actually stands tonight

The active queue in `NORTH_STAR_ROADMAP.md` §5 lists 9 ordered rows. Tracing
each against `git log --oneline main` and the lane directories it produced:

| Row | Roadmap text (2026-07-26) | Actual state tonight (2026-07-28) | Evidence |
|---|---|---|---|
| 1 | Fix the uncertainty model (in flight) | **Landed.** `anchor_sigma_for_bounce` → `BounceAnchorUncertainty`, anisotropic + bias-aware. TT3D depth-ratio 1.65–2.97× → 0.78–1.16×; depth coverage 0.29–0.47 → 0.64–0.86; bias +0.068..0.124 m → −0.003..+0.004 m (bias correction itself stays opt-in). `best_stack` entry `ball.bounce_anchor_uncertainty` = `PENDING`, `do_not_promote` — pickleball-side unverified by design | `runs/lanes/sigma_anisotropic_fix_20260726/REPORT.md`, merged `fix-sigma-20260726` → main |
| 2 | Retire reprojection gating (in flight) | **Landed.** 111 sites classified; 6 changed from reprojection-decides to depth-aware plausibility (`ball_position_plausibility_v1`); 12,866 previously-emitted positions suppressed, 100% from `arc_weak`, 0 from any confident band; exact parity on all 1,203 segment verdicts (0 newly-allowed, 0 newly-suppressed) | `runs/lanes/retire_reprojection_gate_20260726/REPORT.md`, merged `fix-reproj-20260726` → main |
| 3 | Fix the calibration distortion fit | **Landed, twice.** `calib_distortion_fit_20260726` fixed the root cause (net keypoints declared at 0.914 m when every label marks ~0 m — a 0.9 m object-point error, not a distortion gap — plus a dead `intrinsics.dist` seam). `calib_promote_cleanup_20260727` then built the missing selection-pointer mechanism and promoted on **held-out** (leave-one-out) evidence: `pbvision_11min_demo_seed` 2.745→0.177 m (15.5×), `owner_IMG_1605` 2.420→0.107 m (22.7×). Refused burlington/indoor/outdoor/wolverine on the same held-out bar (measured, not omitted). `court.calibration_selection_pointer` = `WIRED_DEFAULT` | `runs/lanes/calib_distortion_fit_20260726/REPORT.md`, `runs/lanes/calib_promote_cleanup_20260727/REPORT.md`, merged `fix-calib-20260726` + `calibpromo-20260727` → main |
| 4 | Sub-frame bounce timing | **Landed, default OFF.** `image_branch_kink_v1`: TT3D bounce error median 0.091→0.031 m (−66%), systematic bias +0.099→−0.008 m (collapses 92%). But the downstream weak-flight-segment trajectory metric did **not** uniformly improve (median −19.8%, p90/p95/max worse) — a measured negative, correctly kept it default OFF. `ball.subframe_bounce_timing` = `PENDING`, `do_not_promote` | `runs/lanes/subframe_bounce_timing_20260727/REPORT.md`, merged `subframe-20260727` → main |
| 5 | Owner bounce labelling at scale (≥150) | **In progress: 25 bounces / ~62 total labels across 4 source-disjoint clips tonight** (wolverine 8, outdoor_webcam_20s 6, burlington 6, indoor 5 — verified by direct read of the label files, not a report). A no-GPU-needed prep lane already staged the next batch | Live read of `runs/lanes/ball_label_tool_20260726/labels/*/ball_human_labels.json`; `runs/lanes/label_clip_prep_20260727/LANE_REPORT.md` |
| 6 | E-v2 scale-up + fine-tune | **Not dispatched. Unblocked tonight — see §4.** This is the plan's #1 action | §4 below |
| 7 | Court + people skeleton closeout | Per roadmap: locked 24-moment review complete, needs a fresh `--force` reproduction; blocked on `gcloud auth login`. Not this plan's focus (owner directed the pivot to events/ball) but action #3 above unblocks it too | `runs/court_skeleton_runtime_20260725/REPORT.md` |
| 8 | Fix association fabrication P0-I | Not touched by anything read tonight; still open per roadmap | `NORTH_STAR_ROADMAP.md` §2.1 P0-I |
| 9 (RF-DETR) | Runs after row 8 | Unchanged, blocked behind row 8 | — |

**Two things not in the original queue at all**, produced 2026-07-27 and worth
knowing before ranking any new ball work:

- **`farfield_extrapolation_20260727`** (merged): investigated whether
  uncorrected distortion explains far-field 3D ball error. **Refuted** —
  plumb-line test on real physical lines gives measured sign **pincushion**,
  opposite of the barrel term the hypothesis needed, and a full distortion
  resweep moves the disputed frame's court-x by 0.086 m total against a
  metre-scale discrepancy. Built `envelope_verdict` tagging instead
  (`within_calibrated_envelope` / `extrapolated` / `far_extrapolated`) — a
  correspondence-radius honesty label, not a fix. `runs/lanes/farfield_extrapolation_20260727/report.json`.
- **`background_ball_20260727`** (merged): confirmed the frozen judge's
  "hidden false positives" are **real pickleballs that are not the ball in
  play** (stray balls on court, adjacent-court balls) — 11/11 indoor, visually
  adjudicated. Then swept every cheap discriminator and refuted all of them:
  detector confidence, heatmap blob radius, image apparent radius, 2D
  position, teleport/continuity structure, and a camera-ray/court-volume gate
  all fail to separate true from false positives (the court-volume gate is
  "sound but useless" — it only fires at 0.0–1.3% of detections). What
  partially works: sequence-level ballistic 3D reasoning, which **already
  exists** in the pipeline. This is independent, fresh confirmation of the
  §2.3 ruling: **the wall is trained contact/event detection, not another
  geometry-only or threshold-based 2D fix.** `runs/lanes/background_ball_20260727/report.json`.

Net effect on this plan: rows 1–4 are engineering-complete and correctly
gated `PENDING`/`do_not_promote` for pickleball because none has independent
pickleball 3D ground truth to score against yet. Row 5 is the cheapest lever
left to accumulate that evidence and is already moving. Row 6 (E-v2) is the
only row where the blocking defect is fully resolved and dispatch is purely a
go/no-go, which is why it leads tomorrow morning.

---

## 2. The ordered plan

### 2a. E-v2 event-head scale-up + pickleball fine-tune

**Why first.** Queue row 6. `SCALE_UP_SPEC.md` (`runs/lanes/event_head_pretrain_20260716/SCALE_UP_SPEC.md`)
shows the 2026-07-16 pretrain was data-starved, not architecture-limited:
label reach 2.4% (1,793/74,546 events), media coverage 18.1%, one window
extracted per row. Three ordered levers compose to ~68× the trainable
windows (226 → ~15,317):

1. **Stage the remaining videos** (22/28 jhong93 + 10/12 OpenTTGames unstaged) — VM-side, h264-direct fetch to avoid the AV1 decode wall that cost 40 min last time.
2. **Multi-window-per-row** — slide extraction (stride ≈ window/2) over full clips instead of one window per row; fix the 226-train/282-val imbalance to ~70/15/15 while there.
3. **Dataloader workers** — the prior run was decode-bound at ~20% GPU util; workers alone should buy 3–4× throughput before any batch-size change.

Cost/GPU-hour table from the spec (A100-40, post-levers): **~2–3h, $2.2–4.5**.
T4 is a legitimate first rung if A100 is unavailable (~7–9h, $1.4–3.6) — do
not wait for a specific accelerator class.

**Also fix, cheap, carried from the 07-16 run** (SCALE_UP_SPEC.md §4):
`eval_event_head.py`'s window mismatch (hardcoded 15-frame vs the
checkpoint's 64-frame context — this alone turned a real 9/9-TP checkpoint
into a false "0 TP" verdict last time) — **already fixed** per the ev2_unblock
lane's blocker-3 closure (`runs/ball_lane_20260723/ev2_unblock/NOTES.md`):
the judge now derives its window from checkpoint provenance and refuses a
mismatched request, byte-pinned by `test_event_head_ev2_judge_control.py`.
Boot-armed rails from VM startup script (not post-RUNNING ssh), AppleDouble
tar hygiene, and a guarded `git rev-parse HEAD` are the other carried fixes —
check they are still in `VM_RUN_PLAN.md` before dispatch.

**Matched-window eval discipline (mandatory, §2.3 no-repeat ruling).** Assert
the eval window against the checkpoint's own `train_manifest.config.window_frames`
at load time — this is the exact defect that manufactured the false "0 TP"
verdict last time and it must never recur silently.

**Plausible firing-rate gate (mandatory, before any anchor is ingested).**
~0.3–1.0 events/s on a real game. The 2026-07-16 zero-shot tennis→pickleball
transfer fired at 7.16 HIT/s — a HIT in 98% of seconds — which is zero
discriminative information, not a weak signal. The fine-tuned checkpoint must
clear the plausible band before anything downstream reads its output as an
anchor.

**Training data.** Fine-tune on the owner's 102 banked labels (61 train / 41
val, gradient-excluded) — `runs/lanes/ball_event_abc_20260720/inputs/owner_102_manifest.json`.
Per the owner-only-asks table (roadmap §5, rank 4): **this pack is DONE — do
not ask for more event labels** unless the first fine-tune shows median
`G_val` ≥ +0.10 but stays < 0.80 macro-F1 (advisory D4). The pb.vision 12
in-domain videos are now fully usage-cleared as training pixels (owner-signed
full rights, 2026-07-20) with their event/ball/court predictions usable only
as an **agreement-filtered teacher** (never ground truth) — the corrected
1,189-row Stage-P manifest already reflects this (0 audio-only rows, down
from 292 in the rejected E0 manifest). Hold 2–3 pb.vision videos out
compare-only so a model that trained on a clip is never scored against
pb.vision on that same clip (§2.3 no-repeat ruling).

**Gate.** Matched-window eval on ≥50 clips with a threshold sweep; plausible
firing rate; the protected 50-row owner seed stays eval-only, never training.

**Kill rule.** If the fine-tune cannot clear the plausible firing-rate band,
do not ingest any anchor from it at any coverage — untyped/implausible audio
or event anchors are permanently barred by §2.3 regardless of how much
coverage they would buy.

### 2b. Typed anchors → sub-frame bounce timing → arc solving with fixed sigma → calibration k1 fit

This chain is **engineering-complete tonight**, in reverse dependency order
from how the roadmap listed it (calibration is the foundation; sub-frame
timing and the sigma fix sit on top of it):

- **Calibration (row 3):** fixed and promoted where the held-out evidence
  supports it (§1 above). The remaining lever is the elevation-parallax term
  that now dominates in/out abstention radius on the promoted demo seed
  (0.4–4.9 m, dwarfing the ~0.02 m a real line call needs) — that is a
  capture-geometry / temporal-resolution problem, not a calibration one, and
  is not solvable by more fitting.
- **Sigma / uncertainty (row 1):** anisotropic, bias-aware, calibration-floored.
  Correctly reports **larger** uncertainty than before, which is the honest
  direction.
- **Sub-frame timing (row 4):** removes 66% of median TT3D bounce error and
  92% of the systematic bias, but the *downstream* weak-segment trajectory
  metric got measurably worse on tails on the same data. **Do not flip this
  to default ON** until that downstream regression is understood — it is a
  real, preserved negative, not a rounding error.
- **Typed anchors (event head, §2a):** this is the piece still missing.
  Everything in this sub-chain refines a bounce anchor's *position*; nothing
  in it decides *whether* a frame is a bounce. That decision is what E-v2
  is for, and §2.3 already rules out any untyped substitute.

**What's actually next here, in order:**

1. Score the sub-frame timing knob and the bias-correction knob against the
   growing owner bounce-label corpus (§2c) as a magnitude sanity check —
   explicitly **not** independent ground truth (owner labels share the same
   ray-plane construction as the solver, so they cannot certify absolute
   accuracy, only relative correction size — this is stated plainly in both
   the sigma and sub-frame reports).
2. Do not attempt to resolve the sub-frame-timing downstream regression by
   tuning against TT3D further — TT3D is table tennis (25fps, 40mm ball,
   ~3.3× the drag of a pickleball); the geometry (a bounce falls between
   frames) transfers, the magnitudes do not. The only way to settle it is
   pickleball ground truth (§2b below and NS-02.1 gold capture).
3. Once E-v2 produces a trustworthy typed-contact signal (§2a), re-run the
   whole anchor→sigma→sub-frame chain end-to-end on real pickleball bounces
   it identifies, and only then consider flipping any of the three PENDING
   `do_not_promote` flags.

**Kill rule (unchanged, §2.3):** no geometry-only 2D→3D retry is a valid next
step here. `background_ball_20260727` independently reconfirmed this
tonight — every cheap geometric/statistical discriminator for even the
*simpler* background-ball problem failed. The wall is trained detection.

### 2c. Owner bounce-labelling at scale (≥150 labels)

**Status tonight:** 25 bounce labels / ~62 total labels across 4 source-disjoint
clips (wolverine 8, outdoor_webcam_20s 6, burlington 6, indoor 5 — verified
directly from `runs/lanes/ball_label_tool_20260726/labels/*/ball_human_labels.json`,
not from a stale report). Bounce is the only label kind with **solved** depth
(ray-plane intersection at the click; the depth slider is locked) — the only
kind that can become truth. Free-flight and near-player stay review-only
human depth estimates forever, by construction (`ball_label_studio.py`
schema fences reject any kind claiming a tier it hasn't earned).

**What unlocks at ≥150.** A source-disjoint held-out split declared *before*
any scoring, giving the first real (if review-only, not NS-02-independent)
read on whether the sigma/sub-frame/calibration chain (§2b) actually reduces
pickleball bounce error — as opposed to the TT3D-only evidence it currently
rests on. It is also the cheapest possible cross-check on E-v2's eventual
bounce-vs-hit classification once that lands.

**Path to more labels without a GPU**, per `runs/lanes/label_clip_prep_20260727/LANE_REPORT.md`:
local BODY is blocked on a missing model checkpoint (not the platform — MPS
and torch are healthy), but the tool **reuses** existing `skeleton3d.json`
artifacts rather than recomputing them:

| Clip | Ready | Bounce candidates | Detection coverage | Est. additional bounces |
|---|---|---|---|---|
| `outdoor_webcam_20s_fullmesh_final` | yes, same calibration as priority-1 (0.101 m floor) | 9 | 51% | 8–16 |
| `burlington_gold_0300_low_steep_corner` | yes, assembled from two runs, cross-checked at 0.083 m median / 0.195 m p90 ankle-XY agreement | 8 | 80% | 6–10 |
| `indoor_doubles_20s_fullmesh_final` | not ready — arc solver produced 0 world_xyz prefills (solver defect, not calibration) | 1 | 40% | 4–10 at much higher effort/bounce |

Ready-now total from the two GPU-free clips: **~14–26 additional bounces**,
which would take the corpus from today's 25 to roughly **39–51** — real
progress toward 150 but not the finish line. A GPU run (outdoor at 60fps
instead of 30fps; indoor's 900-frame encode) is available for **~$5–6, 1.2–1.5h
A100 wall** if the owner wants better bounce-instant timing, but is explicitly
**not required** to keep labelling — one prior GPU attempt lost its output to
a disk-full error mid-fetch (local free space is now 56 GiB, so re-check
headroom before any rerun).

**Gate.** ≥150 bounce labels across ≥4 source-disjoint clips, each carrying
its calibration floor, click sensitivity, and realised sigma; source-disjoint
held-out split declared before scoring.

**Kill rule.** Human labels stay review-only until an independent capture
(NS-02.1) backs them. Never fit a solver on labels it prefilled without
reporting the prefill-corrected fraction (the tool already tracks this via
`origin: fresh|prefill_confirmed|prefill_corrected`).

### 2d. The integration architecture

This is the piece that turns "several correct components" into "the
product." Grounded in what's actually wired tonight, not aspirational:

**What already runs today.** `pipeline_preset=court_skeletons` is the
joint-skeletons-for-most-people-on-most-frames path: 4 players, MHR70
joints, support-foot anchoring, conservative NVZ (kitchen) occupancy, no
meshes/ball/paddle/audio/events/stats/coaching. Measured end-to-end on 6
supported clips: median wall 352.5s for 10–14.8s clips (25.1×–44.7× video
duration), BODY dominates at ~78% of wall
(`runs/court_skeleton_runtime_20260725/REPORT.md`). This is the "joint
skeletons for most people on most frames" half of the owner's target
architecture, and it is the part that is fast enough to run broadly once the
speed work in §2e lands.

**What full meshes need.** `--mesh-coverage-mode ball_aware` already exists
(`RUNBOOK.md`, stage 12 `frames` / stage 13 `body`): physically validated
ball/contact/proximity triggers select which frames get full-mesh BODY
instead of skeleton-only, using `--ball-proximity-m` and
`--high-confidence-swing-floor`. **This is exactly the mechanism the owner's
target architecture needs — meshes only near events/ball contact — and it is
already wired.** What it is missing is a trustworthy contact signal to
trigger on: today's `contact_windows.json` comes from the same
audio/wrist/ball fusion that failed the owner's 50-row spot-check (29/50,
below the ≥47/50 bar) and the event-head checkpoint that has not yet been
scaled up (§2a). **Do not flip mesh scheduling to trust ball-aware triggers
broadly until E-v2 clears its plausible-firing-rate gate** — a bad trigger
here either wastes GPU on empty frames or misses the frames that matter.

**Where `one_world` fusion fits.** `one_world_v1` is wired as a default-OFF
stage at order 185, downstream of `world` (stage 20), and reads
`ball_track.json` + `court_calibration.json`. It is integration progress, not
capability progress — nothing downstream reads its output yet, and its
headline result to date is an **honest refusal**: 0 of 24 declared contacts
confirmed on wolverine (ball >1.2m from every wrist on 22 of them). That
refusal is not necessarily wrong — Track K/L cross-referencing found 19/24 of
those refusals sit on fully-real (non-fabricated) frames — but it means
`one_world` cannot yet be the thing that decides mesh scheduling either. The
sequencing is: **E-v2 typed contacts (§2a) → refined events/arcs (stage
18–19, already explicit timed stages) → `one_world` fusion consumes those
refined artifacts → only then does `one_world` become a candidate input to
mesh-scheduling or trust-band decisions.** Wiring `one_world` as a
mesh-schedule input before that chain is trustworthy would be gating compute
on a stage that currently refuses everything it looks at.

**What the trust bands must say.** Every ball frame already carries
`depth_unvalidated: true` (retire-reprojection-gate lane) and every promoted
calibration carries `authority.class_unchanged: true` (calibration lane) — a
better fit is never a new authority class. The farfield envelope tag
(`within_calibrated_envelope` / `extrapolated` / `far_extrapolated`) should
become a trust-band input once it has a consumer: a ball position tagged
`far_extrapolated` is evidence outside the region any correspondence in the
calibration actually constrains, independent of whether it also passes the
absolute-plausibility bounds. Neither of these is wired into `trust_bands.json`
yet — that wiring is a small, well-scoped follow-up once §2a lands (a trust
band that reads a not-yet-trustworthy contact signal would be premature).

**Cadence, by stage, once this is assembled:**

| Stage | Cadence | Why |
|---|---|---|
| Court/camera calibration | once per clip (v1 static single-lock) | fixed, non-moving camera per product spec |
| TRK + joint skeletons (MHR70, no mesh) | every frame, every visible player | this is the "most people, most frames" tier; already the fast path (§2e target) |
| Ball 2D track + typed event head (E-v2) | every frame | cheap relative to BODY; decides where contact/near-event windows are |
| Full-mesh BODY (`--mesh-coverage-mode ball_aware`) | only frames inside a triggered window | expensive tier, budget-capped by `--target-mesh-frame-budget` |
| Refined events/arcs, `one_world` fusion | once per clip, after BODY | needs BODY+ball+typed contacts together |
| Stats/coaching/manifest | once per clip, last | must not run before the zero-fabrication audit |

**Expected compute per game-minute** cannot be stated as a verified number
tonight — the only measured end-to-end timing (§2e) is on 10–14.8s clips, not
a full game, and it predates the ball-aware mesh triggers actually firing on
real contact evidence. The honest statement is: BODY dominates wall time
today (~78%) at ~62 crops/s, so the product-relevant lever is *how many
frames get scheduled for full mesh*, which is exactly the thing §2a's typed
contacts are supposed to bound tightly. A full game-minute figure should be
the first output of NS-06.1 profiling (§2e) once ball-aware scheduling is
live end to end, not projected in advance.

### 2e. Speed path to the ≤2× video-duration target (NS-06)

Ranked list from `runs/court_skeleton_runtime_20260725/REPORT.md`, with the
measured numbers that justify the ranking (median across 6 clips unless
noted):

| Rank | Lever | Measured basis | Expected saving |
|---|---|---|---|
| 1 | **Persist a warm GPU BODY worker** | model load ≈24.9s + compile ≈33.5s = ~58s per cold job | ~58s/job removed once warm |
| 2 | **Co-locate tracking + BODY on one worker** | upload/download medians 7.4s/6.8s plus decode/wrapper overhead, all currently paid twice | eliminates redundant transfer + decode |
| 3 | **Optimize the tracking bottleneck** | tracking stage costs 77.76–196.07s median 106.30s — the single largest non-BODY stage | must preserve detection/ID/coverage gates; benchmark before selecting |
| 4 | **Share one decode, stream BODY crops** | avoids writing/transferring/rereading `body_frames` | bounded by decode cost, not measured standalone yet |
| 5 | **Reduce non-inference BODY gate/feed work** | array-native gate/feed 15.2–30.6s median 22.1s | profile before cutting; trust checks must survive |
| 6 | **Batch multiple jobs per worker** | amortizes the ~58s cold-start lever 1 already targets | multiplies lever 1's saving across jobs |
| 7 | **Keep mesh disabled in `court_skeletons`** | mesh serialization already correctly skipped (0 bytes, 0s) | preserve, don't reintroduce |
| 8 | **Reduce output transfer size** | current bundles 156.2–231.2 MiB median 193.4 MiB | after correctness closes, not before |

**Constraint that governs all 8:** every lever must preserve the frozen
court/identity/skeleton/placement/kitchen-decision/timebase gates. A faster
run that changes a measured identity or coordinate is rejected outright, per
the report's own framing.

**Relationship to §2a–2d.** Runtime optimization is explicitly *not* the
current accuracy blocker (the report says so directly) — it becomes the
bounded NS-06 lane only after §2a (E-v2) supplies a trustworthy contact
signal for ball-aware mesh scheduling and §2d's cadence table is actually
running end to end. Sequencing NS-06 before that would optimize a mesh
schedule that is currently either off or scheduling on an untrustworthy
signal.

**First fresh benchmark requirement.** The July 25 foot-stabilization and
MHR70 toe-semantic repairs (queue row 7) have not received a complete fresh
GPU timing reproduction at current revision. The first NS-06.1 profiling
pass must record cold **and** warm numbers separately on the current
revision before any lever above is claimed to have saved anything — no
projected saving in the table above is a claimed improvement yet.

---

## 3. "Tomorrow morning" — expanded (see §0 for the compact table)

1. **E-v2 dispatch is the unblock.** Nothing else in this plan is spend-gated
   tonight; this is. Cut a fresh `RUN_COMMIT`, re-run Step-0 at that commit
   (the proof I generated tonight expires; see §4), then dispatch per
   `VM_RUN_PLAN.md`.
2. **Bounce labelling can start immediately, in parallel, for free.** The two
   ready clips in §2c need no GPU and no dispatch decision.
3. **`gcloud auth login`** unblocks both the E-v2 dispatch and the parked
   court/skeleton closeout reproduction (row 7) — one command, do it first.
4. **Do not re-run any of the §2b engineering** (sigma, reprojection retire,
   calibration fit, sub-frame timing) — it is done, merged, and correctly
   gated `PENDING`. The only valid next action on that chain is scoring it
   against a bigger owner label corpus (§2c) or NS-02.1 gold-capture data,
   never more TT3D tuning and never re-deriving what's already measured.
5. **Read the net-keypoint-height handoff** in `runs/manager/inflight_lanes.md`
   before any COURT retrain — it is a live cross-track defect notice, not
   background reading.

---

## 4. Honest constraints

**`VERIFIED=0` is binding on every claim in this document.** Nothing here is
a promotion; every PENDING flag stays PENDING until its named independent
gate passes.

**Owner-shot footage: zero usable rallies, unchanged.** Per
`NORTH_STAR_ROADMAP.md` §2.2 DATA row: total owner-shot pickleball footage is
9.9 seconds, one static pre-serve clip (`IMG_1605.MOV`), zero rallies,
`trainer_forbidden: true`. Every capability in this plan — E-v2 training
media, bounce labels, calibration clips — rides on harvested/competitor video
or the pb.vision grant. This has not changed since 2026-07-26 and nothing in
tonight's research changed it either. The single highest-value owner action
in the whole program remains recording one real game (roadmap §5,
owner-only-asks rank 1), unrelated to and not blocked by anything in this
plan.

**The NS-02.1 gold capture is still the only route to independent 3D ball
ground truth.** Every number in §2b (sigma, sub-frame timing, calibration
floor) is validated on TT3D (table tennis, no LICENSE upstream,
internal-validation-only, never trains, never ships) or on owner bounce
labels that share the solver's own ray-plane construction and therefore
cannot certify absolute accuracy — both reports say this explicitly. Nothing
in this plan substitutes for the gold capture; §2c's label corpus is the
cheapest available cross-check while that capture is pending, not a
replacement for it.

**E-v2 Step-0 training-data gate — verdict: YES, would pass tonight.**

This was verified directly, not inferred from a stale lane report. Tonight
(2026-07-28), from the current `main` checkout (`HEAD=6219514`), I ran the
actual gate script against the actual committed ledger:

```
.venv/bin/python scripts/racketsport/verify_training_inputs.py \
  --inputs runs/ball_lane_20260723/ev2_unblock/training_inputs_ev2.json \
  --ledger runs/manager/data_ledger.json \
  --repo-root /Users/arnavchokshi/Desktop/pickleball \
  --gate-proof <out>
```

Result: **exit 0, `status: PASS`, 4/4 inputs PASS, zero reasons** — the
corrected 1,189-row Stage-P manifest, the frozen T20 init checkpoint, the
owner-102 manifest, and the 40-rally `online_harvest_20260706` media all
cleared, against `data_ledger.json` sha256 `f09e62e1...`.

The evidence chain behind why this is true tonight, when it FAILED as
recently as 2026-07-23:

1. **2026-07-23, early morning:** the E-v2 GPU dispatch was **PREFLIGHT
   BLOCKED** — `verify_training_inputs.py` returned FAIL because
   `event_abc_vm_pull_20260721` was ledger-state `REJECTED` +
   `trainer_forbidden: true`, and `event_abc_inputs_20260720` +
   `online_harvest_20260706` both had `LEDGER_QUEUE_NOT_AUTHORIZED`. No VM
   was created, $0 spent (`runs/manager/gpu_fleet.md`, "2026-07-23 — EVENT
   E-v2 ... PREFLIGHT BLOCKED").
2. **2026-07-23, later that same day (commit `c28951b`, 10:04:43 PDT):** the
   queue-authorization enrichment landed on `main` — all three assets moved
   to `state: READY`/`CONSUMED`, `trainer_forbidden: false`, EVENT-use queued
   disposition. This is *before* the actual VM attempt that afternoon.
3. **2026-07-23, 16:47 PDT:** the real VM dispatch ran. Its Step-0 gate
   **passed** (`gate_proof.json`: `status: PASS`, 8/8 inputs, all 40 rally
   MP4s sha256-verified) — proving the ledger fix was already effective. The
   run then died 10 seconds later on an unrelated code bug: Section 2's
   gate-proof re-check tested `proof.get('pass') is True` against an emitter
   that only ever writes `status: "PASS"` (no `pass` key at all) — a
   deterministic assertion failure on every possible passing proof
   (`runs/lanes/ev2_staging_rootcause_20260724/REPORT.md`).
4. **2026-07-26 (commit `f29145a`):** that assert bug is fixed, plus an
   on-VM log-pull leg was added so the next failure (if any) is diagnosable
   instead of silent.
5. **2026-07-24 (`runs/ball_lane_20260723/ev2_unblock/`):** a dry-run lane
   independently confirmed both fixes with a fresh preflight proof (exit 0,
   4/4 PASS) and a 217-test canonical E-v2 suite pass, and flagged one
   residual (R2): `audit_data_utilization.py`'s enrichment was uncommitted at
   the time. **That residual is also closed** — `git log` shows it landed at
   `4809cce` (2026-07-22, before `c28951b`); running it live tonight gives
   `status: fail` but only on a **generated-view drift** (`DATA_LEDGER.md` is
   stale relative to `data_ledger.json` after later unrelated ledger edits) —
   zero `queue_errors`, zero `ledger_errors`, zero `never_queued`. This
   script is **not** part of the VM_RUN_PLAN.md Step-0 gate (only
   `verify_training_inputs.py` is), so it does not block dispatch, but the
   stale generated view is worth regenerating (`--check-view` failure) before
   anyone reads `DATA_LEDGER.md` as current.

**Caveat for the actual on-VM Step-0** (not re-verified tonight, cloud-side
only): the real `VM_RUN_PLAN.md` Step-0 also hashes `pbvision_gallery_20260719`
media via `INPUT_LOCK.json`'s pinned video-ID→sha256 map — a third asset not
included in the 4-input manifest checked above. That asset's ledger
disposition is `EVENT: CONDITIONAL` (ten non-holdout pixel sources may
train; three compare-only IDs are permanently excluded), unaffected by
anything discussed here, and the `90626da` ledger commit tightened its
`training_allowed_ids` to exactly equal `partitions.train` (a fail-closed
tightening, not a new block). `INPUT_LOCK.json`'s pinned hashes should be
freshness-checked against current cache state at dispatch time — that is a
cloud-side, dispatch-time check the plan cannot substitute for.

**Net: an E-v2 dispatch tonight would pass its Step-0 training-data gate.**
The blocking defect was fixed 2026-07-26; the ledger authorization landed
2026-07-23; both are independently re-verifiable right now with a live
command, which I did. The remaining risk to a dispatch is ordinary cloud-side
risk (capacity, spot preemption, a fresh RUN_COMMIT needing its own Step-0
run) — not a data-safety or authorization gate.

---

## 5. What this plan does not cover

Per the owner's directive, this plan is scoped to events + ball 2D→3D and the
integration architecture that consumes them. It does not re-plan:

- **P0-I (association fabrication)** and the TRK selection layers — queue row
  8, still open, orthogonal to ball/events work but a real trust-contract
  violation that eventually feeds `one_world`'s player-position inputs.
- **RKT (paddle 6DoF)** — parked, unparks only after the NS-02.1 metrology
  capture proves its own ≤1ms sync error, per the roadmap's parked-row entry.
- **Court/skeleton runtime work beyond citing its speed numbers** for §2e —
  the owner indicated that track is being handled separately tonight.

---

## Sources

`NORTH_STAR_ROADMAP.md` §2.2 (BALL, EVENTS/PHYS, FUSION rows), §2.3, §5 rows
1–9; `BALL_TRACKING_PIPELINE.md`; `RUNBOOK.md` ("Reading 3D ball output",
stage order, `--mesh-coverage-mode ball_aware`); `AGENTS.md`;
`runs/lanes/event_head_scaffold_20260716/spec.md`;
`runs/lanes/event_head_pretrain_20260716/SCALE_UP_SPEC.md`;
`runs/lanes/ev2_staging_rootcause_20260724/REPORT.md`;
`runs/ball_lane_20260723/ev2_unblock/{NOTES.md,gate_proof_PREFLIGHT_DRYRUN.json,training_inputs_ev2.json}`;
`runs/manager/gpu_fleet.md` (2026-07-23 E-v2 entries);
`runs/manager/data_ledger.json` (live, tonight);
`runs/lanes/tt3d_external_validation_20260726/report.json`;
`runs/lanes/sigma_anisotropic_fix_20260726/REPORT.md`;
`runs/lanes/retire_reprojection_gate_20260726/REPORT.md`;
`runs/lanes/calib_distortion_fit_20260726/REPORT.md`;
`runs/lanes/calib_promote_cleanup_20260727/REPORT.md`;
`runs/lanes/subframe_bounce_timing_20260727/REPORT.md`;
`runs/lanes/farfield_extrapolation_20260727/report.json`;
`runs/lanes/background_ball_20260727/report.json`;
`runs/lanes/ball_label_tool_20260726/{REPORT.md,labels/*/ball_human_labels.json}`;
`runs/lanes/label_clip_prep_20260727/LANE_REPORT.md`;
`runs/court_skeleton_runtime_20260725/REPORT.md`;
`configs/racketsport/best_stack.json` (revision 18, live read);
`runs/manager/inflight_lanes.md` (2026-07-27 net-keypoint handoff);
`git log --oneline main` (commit-level verification of merge state).
