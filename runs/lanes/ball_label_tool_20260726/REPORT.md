# Ball Label Studio — human 3D ball labels from a single camera

**Lane:** BALL / DATA-1 · **Branch:** `ball-label-tool-20260726` (forked from `d9dbac9`)
**Date:** 2026-07-26 · **VERIFIED=0** — this tool produces human review-only labels, not verified ground truth.

---

## Run it

```bash
.venv/bin/python scripts/racketsport/ball_label_studio.py \
  --run-dir /Users/arnavchokshi/Desktop/pickleball/runs/lanes/w7_critique_20260709/wolv_world/wolverine_mixed_0200_mid_steep_corner \
  --video   /Users/arnavchokshi/Desktop/pickleball/eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4 \
  --out     runs/lanes/ball_label_tool_20260726/labels/wolverine
```

It prints a URL and opens the browser. First launch spends ~1–2 s decoding the clip to
JPEGs; after that, launches are instant and resume exactly where you stopped.

The other verified clip (no `--video` needed, its source resolves on its own):

```bash
.venv/bin/python scripts/racketsport/ball_label_studio.py \
  --run-dir /Users/arnavchokshi/Desktop/pickleball/runs/full_mesh_examples_20260725/outdoor_mesh_final/outdoor_webcam_20s_fullmesh_final \
  --out     runs/lanes/ball_label_tool_20260726/labels/outdoor_webcam
```

CPU-only, localhost-only, no GPU, no cloud, `$0`. The run directory is opened **read-only**
and the tool refuses to start if `--out` would land inside it.

---

## The idea this is built on

A single camera cannot see depth. Clicking the ball gives a **ray**, not a point.

But the task decomposes cleanly:

- **Clicking the ball in the video fixes 2 of 3 degrees of freedom** — the ray.
- **Only depth along that ray is unknown** — exactly one degree of freedom.

So you never place a free 3D point. You click in the video, then move one number.
That is the whole interaction, and it is why a person can do this at all.

---

## What the two panes show

**Left — the video.** The decoded frame at native resolution, with:

- the tracked **player skeletons** projected into the image (blue; red if the pose is flagged implausible),
- the **detector's ball guess** as a dashed yellow circle, labelled "detector guess" so it never reads as truth,
- the other per-frame ball candidates as faint yellow rings,
- the **pipeline prefill** as a dashed purple circle, also clearly not a label,
- your click as a crosshair, and any saved label as a filled ring coloured by kind,
- an **8× magnifier** in the corner that follows the cursor — a pickleball is a handful of
  pixels and it is moving, so this is what keeps clicks accurate. Scroll to zoom up to 16×.

**Right — the live 3D court.** Drawn to regulation scale:

- court surface, all lines, and the net with its real centre (0.864 m) and post (0.914 m) heights,
- the **player skeletons at their tracked 3D positions** — these are the depth reference,
- the **camera position**, so you can see where every ray starts,
- the **ray from your click** as a dashed line with distance ticks every 4 m,
- the **ball marker sliding along that ray**, with a drop line to the court so height is
  readable and a thick bar showing ±1σ of depth uncertainty,
- when the kind is near-player, a line from the ball to the reference joint, labelled with
  the player and joint name,
- your other labels as a faint trail, so an arc's shape is visible as you build it.

Drag to orbit, scroll to zoom.

**Right, below — the top-down minimap.** Plan view of the court with the players, the ray's
ground track, the ball, and a live `x / y / z` readout plus an in-bounds / outside-court call.
This is the fastest way to sanity-check court position.

**Bottom — the timeline.** Bounce candidates (orange), your labels coloured by kind, and every
frame with a ball detection. Click it to seek.

Both views are pinned to the same frame by construction. Frames are pre-decoded to JPEGs
rather than played through an HTML5 `<video>`, because HTML5 seeking is not frame-accurate
and a labelling tool whose frame index can drift from the artifact index is worse than useless.

---

## Keyboard map

Press `?` in the app for this list at any time.

| Key | Action |
|---|---|
| `←` / `→` | step one frame |
| `Shift` + `←` / `→` | step ten frames |
| `Space` | play / pause |
| `B` / `Shift+B` | next / previous detected-bounce candidate |
| `N` / `Shift+N` | next / previous unlabelled frame that has a ball detection |
| `L` / `Shift+L` | next / previous existing label |
| `1` / `2` / `3` | set kind: bounce / near-player / free-flight |
| `K` | cycle label kind |
| click on video | set the ray (2 of 3 DOF) |
| `↑` / `↓` | depth ∓ 0.10 m along the ray |
| `Shift` + `↑` / `↓` | depth ∓ 0.01 m (fine) |
| drag in 3D view | orbit the 3D camera |
| `Enter` | **save the label at this frame** (autosaves immediately) |
| `P` | load the pipeline prefill / detector pixel to correct — `Enter` confirms it |
| `C` | cycle confidence low / medium / high |
| `Backspace` or `Delete` | delete the label at this frame |
| `I` | propose a ballistic arc between the surrounding labels |
| `Shift+I` | accept the proposed arc as free-flight labels |
| `Z` / `Shift+Z` | video zoom in / out |
| `M` | toggle the magnifier |
| `G` | go to frame |
| `?` | toggle the keyboard legend |

A suggested working order: press `B` to jump to a bounce candidate, `P` to load the pipeline's
own pixel, nudge the click if it is off, `Enter`. Those are the labels worth the most, and each
one takes about two seconds.

---

## The three label kinds and their honest accuracy

This is the part that matters most. **The three kinds are not interchangeable and must never
be aggregated into one accuracy number.** The tool records which one every label is, refuses
to let a kind claim a tier it has not earned, and gives each a different uncertainty.

### 1. `bounce` — depth is **solved**, no human depth judgement at all

The ball is resting on the court, so its height is known: `BALL_RADIUS_M = 0.0371` m above the
plane. Depth falls out of ray-plane intersection (`pixel_ray_world` → `intersect_ray_z`), the
same primitives the production arc solver uses. The depth slider is locked. These are the
highest-value labels and the only ones flagged `is_ground_truth_candidate: true`.

Accuracy is bounded by **the calibration, not by you**. The tool measures that bound per clip
by pushing the calibration's own reviewed correspondences back through the exact bounce path
and comparing to their known world positions:

| clip | court-plane error a *perfect* click still inherits (median / p95 / worst) |
|---|---|
| `outdoor_webcam_20s_fullmesh_final` | **0.101 m** / 0.311 m / 0.447 m |
| `wolverine_mixed_0200_mid_steep_corner` | **0.127 m** / — / 0.612 m |
| `indoor_doubles_20s_fullmesh_final` | **0.232 m** / 0.697 m / 0.928 m |

Every bounce label's sigma is floored at that median and combined with a measured click
sensitivity (nudge the click ±2 px and see how far the intersection slides — a grazing ray
near the horizon is genuinely badly conditioned) and the solver's own anchor sigma. Realised
sigma on the outdoor clip: **0.165–0.197 m along the ray**.

Note this is *worse* than `anchor_sigma_for_bounce` alone reports (0.08–0.11 m). The
difference is the calibration residual, which that function does not include. The tool
reports the larger, honest number.

### 2. `near_player` — depth judged against a tracked skeleton. Good, but a human estimate.

Mid-flight, close to a player whose 3D position we already know. You judge the ball's depth
against a shoulder, a head or a paddle hand. The tool finds the tracked joint closest to your
ray, names it in both views, and seeds the depth there.

Sigma starts at a 0.5 m prior and **widens with how far the reference joint actually is from
the ray**, because a "reference" two metres off to the side is barely a reference. It is
capped at the free-flight sigma: having a bad reference can never make you better off than
the honest guess. Realised sigma on the outdoor clip: **0.508–0.534 m**.

The kind is **refused** if no tracked joint lies within 2.5 m of the ray. You are forced to
record it as the free-flight estimate it actually is. Never flagged as a ground-truth candidate.

### 3. `free_flight` — an honest guess, and stored as one

Open space, no reference. Nothing geometric fixes depth. Sigma is **2.0 m along the ray** by
default, deliberately large enough that no downstream consumer can mistake it for a
measurement. Never a ground-truth candidate. The `uncertainty_basis` string on every one of
these says, in words, "Do not treat as ground truth."

### What that looks like in the data

Same clip, one label of each kind, `sigma_xyz_m` (the depth axis here is y):

| kind | `sigma_xyz_m` (m) | σ along ray |
|---|---|---|
| `bounce` | `[0.023, 0.194, 0.040]` | 0.197 m |
| `near_player` | `[0.031, 0.503, 0.071]` | 0.508 m |
| `free_flight` | `[0.091, 1.981, 0.264]` | 2.000 m |

Across-ray precision is excellent in all three cases (~2 cm — that is just the click). **All
the uncertainty is depth**, which is exactly the monocular problem restated. The error
ellipsoid is a cigar pointing at the camera, and the schema stores it that way rather than
flattening it to one scalar.

---

## Prefill, interpolation, autosave

**Prefill.** Where the solver produced a 3D position, `P` loads it — pixel *and* depth — so you
are correcting rather than creating. Every correction is itself a measurement of our error:
the label records `prefill.delta_m` and `delta_px` (a demo bounce moved 0.021 m / 1.98 px).
`P` deliberately **does not save**. A prefill is never promoted to a label without you pressing
`Enter` on it, and the stored `origin` distinguishes `prefill_confirmed` (untouched) from
`prefill_corrected` (you moved it) from `fresh`.

Prefill coverage is per-clip and honest about gaps:

| clip | frames with a solver 3D position | detections | bounce candidates | skeletons |
|---|---|---|---|---|
| `wolverine_mixed_0200_mid_steep_corner` | **300 / 300** | 243 | 10 | 4 players |
| `outdoor_webcam_20s_fullmesh_final` | 189 / 600 | 306 | 9 | 4 players |
| `indoor_doubles_20s_fullmesh_final` | **0 / 600** (solver degraded) | 241 | 1 | 4 players |

Where there is no 3D prefill, `P` still loads the detector's 2D pixel, which is the expensive
half of the click anyway.

**Interpolation.** `I` fits the unique drag-free parabola through the two labels surrounding
the current frame and proposes the frames between them, drawn in both views. It reports the
span, the fitted speed, how far each proposal sits from the detector's own 2D guess, and warns
if the arc passes below the court. `Shift+I` accepts them as `free_flight` labels with
`depth_source: interpolated_arc` and an **inflated** sigma: the drag a parabola neglects is
`rho*Cd*A*v^2/2m`, so the extra term is `0.5*a_drag*T^2`, computed from the actual fitted
speed. An accepted interpolation is always less certain than a directly placed guess, and the
tests assert that.

**Autosave and resume.** Every single label writes the whole artifact atomically (temp file +
`os.replace`), so a crash mid-write cannot truncate an hour of work. A separate session file
records the last frame, and relaunching jumps straight back there.

---

## Output

`ball_human_labels.json` in `--out`. Schema: `docs/racketsport/ball_label_studio_schema.json`,
validated by `threed/racketsport/ball_label_schema.py`. Conventions follow the A-3 metric-3D
contract (`ball_metric3d_contract.py` on `ball-lane-20260723`): the same
`court_netcenter_z_up_m` world frame, required per-axis `sigma_xyz_m` rather than a bare xyz,
fail-closed validation, deterministic serialization with no wall-clock timestamps.

Every label carries: frame, timestamp, clicked pixel, 3D position in metres, kind, accuracy
tier, ground-truth-candidate flag, depth along the ray, the ray itself, depth source, per-axis
and along/across-ray sigma, a plain-English `uncertainty_basis`, human confidence,
`fresh`/`prefill_confirmed`/`prefill_corrected` origin, the prefill it corrected (with deltas),
and the tracked joint a near-player judgement was made against.

The artifact is stamped `verified_ground_truth: false`, `review_only: true`,
`not_ground_truth: true`, and validation **rejects** any payload claiming otherwise.

Raw observations are immutable: the run directory is only ever read, and `open_session` raises
if `--out` resolves inside it.

### The fences that stop an estimate becoming a measurement

These are enforced by the validator, not by convention, and each has a test:

- `accuracy_tier` is derived from `kind` and rejected if a stored value disagrees — a
  free-flight label cannot be hand-edited into `plane_solved`.
- `is_ground_truth_candidate` must equal `kind == "bounce"`.
- `depth_source: ray_plane_intersection` is reserved for bounces.
- A bounce's `z` must equal `BALL_RADIUS_M` exactly.
- `depth_along_ray_m > 0` — a label behind the camera is rejected, not clamped.
- `world_xyz_m` must lie on `ray_origin_m + depth * ray_direction_unit`. **This is the
  sign-error guard.** A flipped parameterisation cannot be persisted even if the fields were
  written independently.
- A non-`fresh` origin must carry the prefill record it used.
- One label per frame, sorted, inside the clip.

---

## Verified against

**`outdoor_webcam_20s_fullmesh_final`** (`runs/full_mesh_examples_20260725/outdoor_mesh_final/`),
600 frames @ 30 fps, 1920x1080, 4 skeleton tracks over all 600 frames, 306 ball detections,
189 solver 3D prefills, 9 bounce candidates, calibration plane residual 0.101 m median.

Also loaded and frame-cached end to end: `wolverine_mixed_0200_mid_steep_corner` (300 frames,
300/300 prefill — the best clip for correcting rather than creating) and
`indoor_doubles_20s_fullmesh_final` (600 frames, solver degraded so no 3D prefill).

A real labelling session was driven through the tool's own API on the outdoor clip, producing
`runs/lanes/ball_label_tool_20260726/demo_labels/outdoor_webcam/ball_human_labels.json`:
15 labels — 4 bounce (`plane_solved`, all 4 ground-truth candidates), 3 near-player
(`player_referenced`), 8 free-flight (`unreferenced_estimate`, from an accepted interpolation),
3 of them `prefill_corrected`.

---

## Test evidence

All commands run from the worktree root with `.venv/bin/python`. Exit codes are real and unpiped.

```
$ .venv/bin/python -m pytest tests/racketsport/test_ball_label_studio.py -q
56 passed in 11.04s
exit=0

$ ruff check threed/racketsport/ball_label_*.py scripts/racketsport/ball_label_studio.py \
      tests/racketsport/test_ball_label_studio.py
All checks passed!
exit=0
```

The 56 tests break down as: 17 geometry (round trip, depth parameterisation, sign guards,
uncertainty decomposition, calibration residuals, ballistic interpolation), 15 contract
(tier/flag forgery, bounce height, off-ray rejection, atomic persistence), 18 studio
integration on a hermetic synthetic run directory (loading, click solving, all three label
kinds, near-player refusal, prefill deltas, interpolation, resume, frame cache), 5 server
end-to-end (page serve, JPEG serve, ray, save, autosave-on-disk, delete, token rejection,
bad-label rejection), and 1 real-clip check that skips if the corpus is absent.

### Repo structure checks (AGENTS.md)

```
$ .venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .
exit=0
$ .venv/bin/python scripts/racketsport/audit_dead_code.py --root .
exit=0
$ python3 scripts/racketsport/audit_storage_policy.py --root . --json
exit=1
```

`audit_storage_policy` **fails, and the failure is pre-existing and not attributable to this
change.** It flags `unknown_large_tracked_files` under `runs/lanes/holdout_eval_20260721/...`
and `missing_allowed_large_untracked_source_files` under `cvat_upload/court_diversity_20260712/...`.
Filtering its output for `ball_label` returns an empty list. Nothing this change adds is large,
tracked, or outside policy; the extracted frame caches live under `runs/` and are gitignored.

The new CLI is registered in the scaffold index with a related test, a direct CLI reference
test and a matching JSON schema:

```json
{"command_path": "scripts/racketsport/ball_label_studio.py", "category": "label",
 "workstream": "BALL", "task_prefix": "DATA-1",
 "related_test": "tests/racketsport/test_ball_label_studio.py",
 "direct_cli_reference_test": "tests/racketsport/test_ball_label_studio.py",
 "matching_schema": "docs/racketsport/ball_label_studio_schema.json"}
```

Index summary: `tool_count 324, missing_related_tests 0, missing_direct_cli_reference_tests 0`.

`tests/racketsport/test_scaffold_tool_index.py::test_real_scaffold_tool_index_matches_checked_in_schema`
fails on `category_counts["unknown"] == 7`. This is also pre-existing: stashing the one-line
`list_scaffold_tools.py` change and rerunning reproduces the identical failure, and the seven
uncategorised tools (`abc_decision_gate`, `apply_event_sequence_dp`, `build_abc_arm_manifests`,
`build_court_v31_protocol`, `build_data_inventory`, `build_owner_event_manifest`,
`verify_training_inputs`) are all unrelated to this lane.

### Browser check

Chrome headless and the installed Playwright are both unusable in this environment (Chrome
exits 137 under the sandbox; the Playwright driver fails to start on Python 3.14), and
installing browser tooling was out of scope. Instead the page's own script is executed in
**Node against a stub DOM and canvas, talking to a real running server** —
`tests/racketsport/fixtures/ball_label_studio_page_harness.mjs`, wrapped by
`test_the_served_page_boots_renders_and_saves_in_a_headless_dom`. Against the real outdoor clip:

```json
{"ok": true, "errors": [], "console_errors": [],
 "clip_id": "outdoor_webcam_20s_fullmesh_final", "frame_count": 600,
 "skeleton_frames": 600, "bounce_candidates": 9,
 "canvas_calls": {"fillRect": 4158, "stroke": 1707, "fill": 2153,
                  "drawImage": 18, "fillText": 312, "arc": 2295}}
```

That confirms: boot against real `/api/state`, all four canvases drawing real skeletons and
markers, frame stepping, kind selection, depth nudging, and a full save then verify then delete
round trip through the real server. It also re-checks **in the browser's own code** that
`origin + t * direction` is metres from the camera, which is the one piece of geometry the page
owns. The page script additionally passes `node --check`.

### Manual smoke test (2 minutes)

1. Run the launch command above; the browser opens.
2. Header shows the clip, `frame 0 / 299`, and zero counts. Left pane shows the video with blue
   skeletons over the players; right pane shows the court, net and the same skeletons in 3D.
3. Press `B`. Both panes jump to the first bounce candidate, together.
4. Press `P`. The crosshair lands on the pipeline's pixel, the kind chip goes green (Bounce),
   the depth slider is disabled (solved), and the 3D pane draws the ray with the ball on the court.
5. Press `Enter`. The status line turns green with the label count; the timeline gains a green tick.
6. Press `3`, then `↑` a few times. The slider unlocks, the marker slides out along the ray in
   the 3D view and on the minimap, the sigma bar grows, and the `xyz` readout changes — while
   the crosshair in the video pane does **not** move, which is the ray property made visible.
7. Press `Delete`, then quit and relaunch. It resumes at the same frame with the same labels.

---

## Honest limitations

- **VERIFIED=0.** These are human review-only labels. Only bounce labels have a solved depth.
  Nothing here has been gated, and no accuracy claim is promoted.
- **Bounce accuracy is capped by the calibration, and the cap is large** — 0.10 m median on the
  best verified clip, 0.23 m median and 0.93 m worst-case on the indoor one. Labelling cannot
  make a clip's calibration better. If sub-decimetre bounce truth is needed, the calibration is
  the thing to fix first, not the labelling.
- **Distortion is not undistorted before ray casting.** This matches the production solver
  exactly (`ball_arc_solver` also feeds raw pixels to `pixel_ray_world`), which is what makes
  labels directly comparable to solver output. Measured cost on the indoor clip, which has real
  k1/k2: undistorting first would move the median plane residual 0.232 m to 0.199 m. It is not
  the dominant error term, but it is not zero either.
- **Near-player and free-flight sigmas are priors, not measurements.** 0.5 m and 2.0 m are
  honest engineering judgements about human depth perception. Nobody has measured how well this
  particular owner resolves depth against a skeleton. A good follow-up: have them label bounces
  as if they were free-flight and compare against the solved answer — that would turn the prior
  into a measurement. The CLI exposes `--near-player-sigma-m` and `--free-flight-sigma-m` so the
  numbers can be updated when that evidence exists.
- **Skeletons are themselves estimates.** A near-player label inherits the error of the tracked
  3D pose it was judged against, which is not included in its sigma. The tool surfaces the pose's
  own `skeleton_implausible` flag (red skeleton) but does not propagate pose uncertainty.
- **Interpolation neglects drag** and its sigma inflation is a first-order correction, valid over
  short spans. Over a long span the parabola will be visibly wrong; the tool shows the fitted
  speed, the span, and the detector residual so that is obvious, but it does not refuse.
- **One label per frame.** No multi-ball, and no explicit "ball not visible" marker — an
  unlabelled frame is currently ambiguous between "not looked at" and "not visible".
- **`ball_metric3d_contract.py` was matched by convention, not imported.** It lives on
  `ball-lane-20260723`, not on this branch. If the two lanes merge, the honest next step is to
  express `BallLabel` as a `GroundTruthObservation` with `cameras_used: ["monocular_human"]`
  and quality flags carrying the tier — but only bounce labels should make that crossing.
- The indoor clip's solver output is `degraded` with zero 3D positions, so it offers no 3D
  prefill. The tool works there, but labelling it is slower.
