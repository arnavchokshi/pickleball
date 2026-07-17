# Lane spec — trackI_placefuse_20260716
# Track I: per-player court-frame trajectory refiner (people-placement fusion)

You are a Codex implementation lane (gpt-5.6-sol, high). You self-verify and return a
structured report; the Track I manager rules on the report, never the transcript.

## 0. HARD RULES

- NO git branches, NO commits (owner joint-commit rule; the manager commits fence-only after ruling).
- Read first: `NORTH_STAR_ROADMAP.md` §2.2 (BODY/TRK rows), §3.1 reuse contract (especially the
  BODY row "BODY translation is not identity truth" and the Global-fusion row "raw observations
  immutable"), NS-04.4 surface priors ("Never snap ... ankle centers to the floor"), §6 standing rules.
- The 4 internal clips (burlington / outdoor / wolverine / img1605) are EVAL-ONLY evidence here:
  you consume banked pipeline artifacts; you never touch, create, or consume label files; no training.
- No per-clip hyperparameter tuning (threshold shopping). ONE global config for the candidate.
- Honest reporting. A number that misses its bar is reported as the number it is.
- Raw artifacts are IMMUTABLE: you never rewrite `skeleton3d.json`, `tracks.json`,
  `foot_contact_phases.json`, or anything inside `runs/lanes/w3_freshworlds_20260707/` or
  `runs/lanes/w4_footattr_fix_20260707/`. Other lanes' run dirs are READ-ONLY evidence.
  Your refined output is a NEW artifact under YOUR lane dir.
- NEVER emit a .patch file. Any change you believe is needed in a fenced file (e.g. stage wiring in
  `scripts/racketsport/process_video.py`, owned by Track C) goes in the report as inline diff hunks
  + rationale; Track C re-derives.
- Run the WIDE blast-radius suite (`MPLBACKEND=Agg .venv/bin/python -m pytest tests -q`), not a
  hand-picked subset. Known sandbox socket/MPS failures must be identified as such with evidence.
- Every new CLI ships its direct-CLI reference test same-lane and must appear in
  `scripts/racketsport/list_scaffold_tools.py` output (see file-ownership note on that file below).
- All lane artifacts under `runs/lanes/trackI_placefuse_20260716/`.
- MPLBACKEND=Agg for anything matplotlib-adjacent. Absolute paths in all reported evidence.

## 1. EXPLICIT FILE OWNERSHIP (must stay disjoint from live lanes)

YOU OWN (all NEW files):
- `threed/racketsport/placement_trajectory_refine.py`  (the refiner module — NEW)
- `scripts/racketsport/validate_placement_slide.py`    (frozen scorer CLI — NEW)
- `scripts/racketsport/build_refined_placement.py`     (refiner CLI — NEW)
- `tests/racketsport/test_placement_trajectory_refine.py` (NEW)
- `tests/racketsport/test_placement_refine_clis.py`    (direct-CLI reference tests — NEW)
- `runs/lanes/trackI_placefuse_20260716/**`
- `scripts/racketsport/list_scaffold_tools.py` — ADDITIVE dict entries ONLY (TASK_HINTS /
  RELATED_TEST_OVERRIDES rows for your two CLIs). A concurrent lane (Track G) also appends entries
  to this file: make this your LAST code edit, re-read the file fresh immediately before editing,
  and keep the diff to added lines only. If your chosen CLI verb prefixes would not be indexed by
  its PREFIXES list, rename your CLIs to indexed prefixes rather than widening PREFIXES.

READ-ONLY (import, never edit): `threed/racketsport/placement.py` (its typed-coordinates API was
just adopted — preserve it by importing, not modifying), `foot_contact.py`, `worldhmr.py`,
`body_grounding_quality.py`, `body_grounding_refine.py`, `coordinates.py`, `schemas/__init__.py`.
FORBIDDEN entirely: `scripts/racketsport/process_video.py` (Track C live), `orchestrator.py`
(Track C live), `threed/racketsport/ball_arc_*` (Track A), `threed/racketsport/event_head/**` and
event_head CLIs (Track G), `ios/**`, `web/**`, `configs/racketsport/best_stack.json` (propose hunks
only), all root .md files.

## 2. OBJECTIVE

Owner directive (verbatim intent): SAM-3D-Body meshes are already accurate — the gap is PLACEMENT.
"We just need to focus on putting the people in the correct spot... we don't fully trust single
things, but we use all info together to produce better results, combining things we are most
confident in."

Build a per-player COURT-FRAME trajectory refiner that fuses, confidence-weighted:
- TRK court footpoints (`tracks.json` per-frame `world_xy` / `fused_world_xy`, `conf`, bbox),
- SAM-3D-Body root + foot trajectories (`skeleton3d.json`: `joints_world` 70 joints court_Z0,
  `transl_world`, `joint_conf`; root-relative accuracy is the trusted part — 59.7mm root-rel /
  39.9mm PA externally; world grounding is the weak part),
- foot-lock / plant-phase evidence (skeleton-direct accepted contact phases: while a foot is
  planted its court position should be near-stationary) as SOFT anchor terms,
- court-plane priors on grounded soles as SOFT bounded terms (NS-04.4: never snap),
- temporal smoothness (e.g. second-difference penalty) with robust losses on every evidence term.

Recommended shape (you may deviate with justification): per player, solve a robust weighted
least-squares / factor-graph style problem for a per-frame RIGID court-XY correction applied to the
whole skeleton (root-relative pose untouched — that is the trusted signal; do not re-pose joints).
Z corrections only if bounded, soft, and justified; report their effect on producer phase
acceptance separately. No raw averaging of modalities — documented bounded, confidence-weighted
combination (NS-01.7 rule). Key insight from the artifacts: each skeleton frame carries
`confidence_provenance.grounding_anchor_source = "placement_track_world_xy"` — BODY world placement
is per-frame anchored to the TRK placement point, so per-frame anchor jitter propagates directly
into planted-foot slide. Your refiner exists to replace per-frame anchoring with a
whole-trajectory, plant-aware, confidence-weighted estimate.

OUTPUT ARTIFACT (NEW, per card, under your lane dir):
`placement_trajectory_refined.json` containing per player per frame: refined root (`transl_world`),
refined foot positions (or the rigid correction vector + convention to apply it), 2x2 (or 3x3)
covariance, per-frame per-term provenance (which evidence contributed at what weight, which frames
were plant-anchored), correction magnitude, and a top-level provenance block (inputs w/ sha256,
config, code version, coordinate space `court_Z0` declared explicitly, preview_band=true,
VERIFIED=0). Raw inputs immutable. Also write `SCHEMA.md` in the lane dir documenting every field —
Track K (global fusion) consumes this schema.

## 3. EVIDENCE TO READ FIRST

- `runs/lanes/w4_footattr_fix_20260707/report_r2.json` — the baseline definition. Baseline
  skeleton-direct accepted-phase max slide (the frozen gate metric, `max_foot_lock_slide_m`
  family): burlington 0.03455377233204998, outdoor 0.033610376308433464,
  wolverine 0.02080867319073746, img1605 0.048380190412025174. 3/4 breach the frozen 0.030 bar.
- `runs/lanes/w4_footattr_fix_20260707/skeleton_direct/<clip>/foot_contact_phases.json` — the
  accepted skeleton-direct phases (Design A: producer may only reject on independent quality —
  confidence, penetration, agreement — NEVER on a slide threshold; that is circular and forbidden).
- Banked complete pipeline outputs (READ-ONLY), nested `<clip>/<card>/<card>/`:
  `runs/lanes/w3_freshworlds_20260707/burlington/burlington_gold_0300_low_steep_corner/burlington_gold_0300_low_steep_corner/`
  `runs/lanes/w3_freshworlds_20260707/outdoor/outdoor_webcam_iynbd_1500_long_high_baseline/outdoor_webcam_iynbd_1500_long_high_baseline/`
  `runs/lanes/w3_freshworlds_20260707/wolverine/wolverine_mixed_0200_mid_steep_corner/wolverine_mixed_0200_mid_steep_corner/`
  `runs/lanes/w3_freshworlds_20260707/img1605/owner_IMG_1605_8a193402780b/owner_IMG_1605_8a193402780b/`
  Each has `tracks.json`, `placement.json`, `court_calibration.json`, `skeleton3d.json`,
  `sam3d_keypoints_2d.json`, `smpl_motion.json`, `body_mesh_index/`, `net_plane.json`.
- `threed/racketsport/foot_contact.py` — `build_body_skeleton_foot_contact_phases` (~line 333) is
  the frozen phase producer; IMPORT it, never copy/fork its logic.
- `threed/racketsport/worldhmr.py` lines ~292-1060 — how `max_foot_lock_slide_m` /
  `foot_lock_slide_p95_m` / gate stream are computed (`compute_body_skeleton_and_metrics`,
  `_apply_refined_stance_phase_lock_and_pin`, `_contact_gate_stream_for_skeleton3d`).
- `threed/racketsport/placement.py` — `PlacementConfig` foot keypoint index maps
  (`SAM3D_FOOT_KEYPOINT_INDICES`, `NATIVE2D_FOOT_NAMES`), `inverse_covariance_fuse`, and the typed
  coordinate usage pattern. `threed/racketsport/coordinates.py` — the typed coordinate API; use it
  for EVERY image<->court transform; declare coordinate space + distortion state in your artifact.
- `runs/lanes/ns014_p22residual_20260709/REPORT.md` — decode-residual context (why root-relative is
  trusted and the postchain is where error lives).

## 4. ORDERED WORK PLAN

STEP 1 — FROZEN SCORER FIRST (`validate_placement_slide.py`).
One deterministic CLI that, given a skeleton3d-shaped payload (file path) + phases source, computes:
 (a) skeleton-direct accepted-phase max slide + p95 + per-phase table (via the imported frozen
     producer applied to that payload),
 (b) slide on an explicit FROZEN phase-window list supplied as input (so refined trajectories can
     be scored on baseline windows without re-selection),
 (c) reprojection metrics: project per-frame foot midpoint + root through the card's calibration
     (typed API) into image px; report median/p95 px distance vs 2D evidence
     (`sam3d_keypoints_2d.json` foot keypoints where present, else tracks bbox bottom-center),
 (d) TRK-vs-BODY footpoint disagreement: per-frame court-XY distance between the TRK track point
     (`world_xy`/`fused_world_xy`) and the BODY grounded-foot court point; median/p95 per clip.
ACCEPTANCE 1: scorer on the UNMODIFIED banked artifacts reproduces the four baseline numbers above
to =1e-9 relative (same inputs, same producer code path). If it cannot, STOP — report the exact
divergence and why; do not proceed to the refiner on top of an unproven scorer.
Save the full baseline table (a)-(d) for all 4 cards to
`runs/lanes/trackI_placefuse_20260716/baseline_metrics.json` BEFORE any refiner work.

STEP 2 — REFINER (`placement_trajectory_refine.py` + `build_refined_placement.py`).
Implement the fusion described in §2. Requirements:
- typed, fail-closed errors on missing/malformed inputs (no silent defaults);
- robust losses (e.g. Huber) on TRK terms — a wrong footpoint must not drag the trajectory;
- confidence weighting from real fields (`conf`, `joint_conf`, phase confidence), documented;
- plant-phase stationarity as SOFT anchors (finite weight, bounded correction), never a hard
  equality, never a projection/snap;
- covariance out (from the normal-equations inverse or a documented bounded approximation);
- deterministic: same command twice => byte-identical artifact (report md5 of both runs).

STEP 3 — SCORE (same scorer, both arms, all 4 cards) into
`runs/lanes/trackI_placefuse_20260716/fused_metrics.json`:
- Arm A (primary): producer REBUILT phases on the refined payload -> accepted-phase max slide.
- Arm B (anti-gaming): slide of the refined trajectory measured on the FROZEN baseline windows.
Also recompute (c) reprojection and (d) disagreement on the refined payload.

STEP 4 — TESTS. Synthetic unit tests that gate the capability (no logging-only tests):
- synthetic trajectory with known jitter + known plants: refiner reduces plant-window slide below a
  stated bound while a clean high-confidence TRK-agreeing segment moves < a small epsilon;
- outlier footpoint robustness (one 2m-wrong TRK point does not move the plant);
- no-snap proof: with court-plane prior active, sole Z is NOT exactly 0 post-refine unless it was 0
  pre-refine (assert no clamping);
- fail-closed on missing calibration / empty phases / NaN confidences (typed exceptions);
- determinism test;
- direct-CLI reference tests for both CLIs.

STEP 5 — SENSITIVITY. Re-run scoring at 0.5x and 2x on the 2-3 dominant weights (one at a time).
Report the slide table for each. The headline result must not be a knife-edge.

STEP 6 — scaffold-index registration (last code edit, per §1), wide suite, report.

## 5. ACCEPTANCE NUMBERS (report each; the manager rules)

1. Scorer baseline reproduction: 4/4 clips match the w4 numbers to =1e-9 rel. HARD GATE to proceed.
2. PRIMARY: Arm A accepted-phase max slide =0.030 on 4/4 clips (baseline: 3/4 breach). If <4/4,
   report per-clip honest values + diagnosis; =3/4 with no clip worse than baseline is still a
   reportable improvement, NOT a pass of the frozen bar.
3. Arm B (frozen windows): fused slide strictly below baseline slide per clip (max and per-phase
   median). A candidate that wins Arm A but not Arm B is presumed phase-selection gaming — FAIL.
4. Phase preservation: Arm A accepted phase count within +/-20% of baseline per clip; every change
   itemized by producer rejection reason.
5. Reprojection non-degradation: per clip, fused median and p95 px error each worsen by no more
   than max(1.0px, 5%) vs baseline. Improvement is a bonus, not required.
6. Disagreement diagnostic: baseline TRK-vs-BODY median/p95 per clip reported; post-fusion
   fused-vs-TRK and fused-vs-BODY residuals reported (fused should sit between the sources,
   weighted toward the higher-confidence one; explain any clip where it does not).
7. No-snap evidence: correction-magnitude distribution (median/p95/max, per clip); assert-backed
   proof no sole-Z clamping; NS-04.4 kill rule explicitly honored.
8. Sensitivity table present; stated conclusion about robustness.
9. Determinism md5 pair; typed-failure tests green.
10. Wide suite: `MPLBACKEND=Agg .venv/bin/python -m pytest tests -q` with REAL unpiped exit code;
    failures>0 require proof each failure is pre-existing/sandbox-known (name them).

KILL CRITERIA: If after honest effort Arm B cannot beat baseline on >=3/4 clips without breaking
acceptance 5, stop tuning and report FAIL with a written diagnosis of where the slide actually
comes from (e.g. root-relative foot motion inside the phase rather than anchor jitter — quantify
the split: slide of root-relative-only foot vs slide of anchor-only). That diagnosis is a valid
deliverable.

## 6. OUT OF SCOPE (state in report, do not attempt)

- pb.vision 11-min demo video: NO pipeline/BODY artifacts exist for it and BODY requires a GPU run
  (fleet priority is elsewhere today). Report as an honest limitation + what a follow-up needs.
- Stage wiring into `process_video.py` (Track C owns it): produce the exact inline wiring hunk
  (new optional stage `placement_trajectory_refine` after `grounding_refine`, reading banked
  artifacts, writing the new artifact) in your report for Track C to re-derive. Do NOT apply it.
- Any GPU/BODY re-run. All work is CPU-local from banked artifacts.

## 7. MANDATORY STRUCTURED REPORT (report.json via the output schema)

- `objective_result`: PASS / FAIL / PARTIAL strictly vs §5 numbers (acceptance 2 governs PASS).
- Baseline table AND fused table (per clip: Arm A slide, Arm B slide, phase counts, reprojection
  median/p95, disagreement median/p95, correction magnitudes).
- `full_suite`: wide-suite passed/failed/skipped + real exit code + pre-existing-failure proofs.
- HONEST ISSUES list (including the inherent one: hyperparameters chosen on the same 4 internal
  cards => scoped evidence only, VERIFIED=0 stands; refined artifacts are preview-band).
- Artifacts list (absolute paths). Changes list (file:line).
- One readable paragraph: what the fusion actually does, in plain language.
- BEST-STACK DELTA: expected (b)-as-proposal — propose the PENDING `best_stack.json` entry for the
  refiner (revision-bumped, preview-band, do_not_promote) as an INLINE HUNK in the report; do NOT
  edit the file. If you conclude (c) no delta, say why.
- Dated bullet for the manager's ledger row.
- The inline `process_video.py` wiring hunk for Track C (see §6).
