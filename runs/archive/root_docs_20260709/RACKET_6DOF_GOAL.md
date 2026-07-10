# Racket 6-DOF Goal

Date opened: 2026-07-05
Status: PHASE 1 FINAL_V3 LANDED 2026-07-05 as render-only internal-val. VERIFIED=0 unchanged. RKT board row: SCAFFOLD.
Render band for everything this goal ships: ESTIMATED/preview, never silent truth.

## Goal (owner directive, 2026-07-05)

Whenever possible, the 3D virtual world renders a **full 6-DOF (position + orientation) paddle**
for the player holding it, as accurate as achievable from single-camera video — driven primarily
by the player's **wrist/hand** and by **ball direction** at contact. When evidence is absent the
paddle degrades honestly (band drop or absence), never a silent fake.

## Standing constraints (repo truth — none reopened by this goal)

1. **"Paddle rectangle-to-6DoF promotion" stays killed** (MASTER_PLAN non-negotiables).
   Box-only poses are IPPE-ambiguous and stay suppressed in the world
   (proof: `runs/rkt_paddle_lane_20260704T204142Z_wolverine/RKT_PADDLE_BLOCKER_REPORT.md`,
   544/544 ambiguous preview frames, world suppression proof, 83+8 tests green).
   This goal must add **new evidence channels** (wrist/hand anchor, ball reflection, mask fit,
   temporal grip rigidity), not re-derive pose from rectangles.
2. **Promotion gate unchanged**: face-angle/contact-point error vs true-corner/reference GT
   (owner 4-marker capture still pending). Until then RKT never promotes; outputs are render-only.
3. CVAT `paddle` rectangles (Burlington 13 / Wolverine 14 / Outdoor 17 tracks; Indoor 0) are
   **scoring-only forever** — never solver input on the clip being scored. Outdoor/Indoor stay
   strict held-out: any Outdoor scoring needs a pre-registered `heldout_eval_ledger.md` row first.
4. File fencing (2026-07-05): `ball_i1_default_integration_20260705` (LIVE) owns
   `process_video.py`, `virtual_world.py`, `web/replay`; the speed lane may instrument
   `process_video.py`. Racket lanes ship **new files only** + deferred integration patches.

## Prior evidence (do not re-derive)

- **Detection**: zero-shot GroundingDINO-tiny+SAM2-tiny: recall 0.976 @IoU0.25 on Outdoor but
  precision 0.0078 unfiltered (ledger RKT-1) — a wrist gate is the obvious missing filter.
  Trained YOLO26s (`runs/rkt_train_20260702T072800Z/`): external AP50 0.74, owner-clip pooled
  AP50 0.27 (Outdoor 0.50 / Burlington 0.13) — domain gap, usable as weak evidence only.
- **Wrist**: strongest joint chain in the system — SAM3D wrist bone lock landed (lower-arm CV 0.0,
  swing-peak timing 0-frame delta); placement lane closed with 3/4 clips all-green worlds.
- **Ball**: arc solver ships measured-grade arcs (P 0.88–0.90, recall ~0.5, 0 teleports);
  contact events exist via event fusion; ball chain becoming a default pipeline stage (live lane).
- **Scaffold already built**: `racket_true_corners.py`, preview IPPE pose + reprojection
  diagnostics, promotion audit, readiness gates, review crop sheets, candidate overlays,
  `racket_dataset_schema.json` (source types: racketvision | synthetic_blenderproc | aruco_gt).

## Architecture theory (Fable 2026-07-05 — to be validated by the research wave)

**Parameterization (the core idea): the paddle is rigidly gripped.** Solve
`X_paddle(t) = W_hand(t) ∘ G` where `W_hand(t)` is the wrist/hand frame from the SAM-3D skeleton
and `G` is a **grip transform (hand→paddle) held constant per grip segment**, plus a small
per-frame deviation. Hundreds of frames then share one 6-param unknown, so sparse strong evidence
(contacts, clean masks) locks `G`, and **wrist alone carries a full 6-DOF paddle through frames
with no direct paddle evidence** — exactly the owner's "whenever possible".

**Evidence channels feeding one robust solver (per player, per rally):**
1. **Hand anchor** — wrist position + forearm axis; pronation/supination (the DOF that sets the
   face normal) from MHR70 hand joints if emitted, else from a 2D hand-pose model on wrist crops
   (open question Q1 below).
2. **Ball reflection at contact** — at fused contact events, with ball velocity in/out (3D arcs
   when available, 2D projected constraint otherwise) and paddle velocity from wrist motion:
   face normal ≈ bisector of relative velocity change (soft cone, spin-tolerant); the contact
   point must lie on the paddle face. Strongest orientation evidence in the whole system, exactly
   at the frames coaches care about.
3. **Mask/box reprojection** — project a parametric paddle model (face ellipse + handle) through
   the clip calibration; fit to wrist-gated SAM2 masks (or detector boxes as weak fallback).
4. **Temporal rigidity** — `G` constant within grip segments (regrip = detected residual break),
   smooth small deviations; two-handed/hand-switch handling.

**Trust bands (observability-driven, per frame):** `contact_locked` (± a few frames of a
contact where reflection+mask agree) > `mask_fitted` > `grip_extrapolated` (wrist+known G only)
> absent (no wrist / excluded player). Bands ride the existing trust-band machinery.

**Honest validation without corner GT (internal iteration on Burlington/Wolverine only):**
- Leave-one-channel-out: fit without masks on validation frames → mask IoU there;
  fit without half the contacts → reflection residual on the held-out contacts.
- 2D scoring vs CVAT paddle rectangles (scoring-only) on internal clips.
- Temporal jitter/rigidity stats + overlay video + viewer visual review kit.
- Optional later: synthetic BlenderProc paddle sequences with exact GT to calibrate solver error
  bars (schema already anticipates this); owner 4-marker capture remains the only promotion path.

## Open questions the research wave must answer (evidence, not opinions)

- **Q1**: Do emitted skeleton3d/MHR70 artifacts contain hand/finger joints (which names/indices),
  and at what per-frame availability/confidence on the 4 final run dirs? If absent: what do the
  raw SAM-3D model outputs expose that we currently discard, and at what cost to emit?
- **Q2**: Per clip: how many fused contact events, and for what fraction is ball velocity
  (3D arc or clean 2D) available in a ±5-frame window?
- **Q3**: What paddle mask/box artifacts exist per clip today (paths, coverage %, quality), and
  what would wrist-cropped re-detection cost (local MPS vs A100)?
- **Q4**: External SOTA worth stealing for hand-frame recovery on wrist crops (HaMeR/WiLoR-class)
  and racket/paddle 6-DoF or reflection-based estimation (incl. what "racketvision" refers to);
  licenses recorded verbatim (internal-R&D use OK per owner ruling 2026-07-04).

## Lane plan

- Lane home: `runs/lanes/racket_6dof_20260705/` (STATUS.md = trail).
- R1 (Codex, repo, read-only): quantify Q1–Q3 on the four final placement run dirs.
- R2 (Codex, web research): Q4 survey with license table.
- After Fable rules on R1/R2: implementation lanes (new files only:
  `threed/racketsport/paddle_pose_6dof.py` + CLI + tests is the working plan), then
  deferred-patch world/viewer integration once the ball lane lands its virtual_world changes.

## Log

- 2026-07-05: goal opened (owner directive); repo scout + goal doc; research wave dispatched.
- 2026-07-05 ~19:00Z: R1+R2 landed. Fingers 100% available all clips; proxy baseline weak
  (IoU 0.11/0.03, jitter 23-53 deg/f); ball channel empty in vp1 run dirs; WiLoR = top external
  hand-frame candidate; reflection = soft-cone only. Ruling: phase 1 = grip-transform fusion,
  CPU-only, no new models.
- 2026-07-05 ~20:30Z: I1 LANDED (Sonnet legs after Codex quota exhaustion).
  Initial report numbers were superseded by final_v3 acceptance evidence.
- 2026-07-05 final_v3: phase-1 fused estimator accepted for render-only
  review, not RKT promotion. Evidence:
  `runs/lanes/racket_6dof_20260705/i1_fused_estimator/acceptance_record_v2.json`
  and `runs/lanes/racket_6dof_20260705/STATUS.md`. Wolverine internal-val
  IoU is 0.2356, median center error 23.4px, rotation jitter max p95 27.9 deg;
  Burlington IoU is 0.3424, median center error 13.4px. Coverage is 100% in the
  scored clips, bands are 100% `palm_fitted`, and `contact_locked` remains
  dormant until real 3D contacts exist. The final teleport census is 29 -> 0
  undeclared one-frame jumps >0.35m across the four clips; five declared switch
  jumps remain. The Wolverine IoU is below the earlier 0.24 leg-2 floor, so do
  not repeat "all acceptance bars met" as a global statement. RKT stays
  SCAFFOLD and render-only until true-corner/reference GT passes.
  Phase 2 remains queued: P2a wrist-gated masks, P2b WiLoR pronation, P2c
  IMG_1605 GPU ball track (30 audio onsets = first real reflection test bed).
