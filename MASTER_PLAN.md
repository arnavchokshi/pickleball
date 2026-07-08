# Pickleball Master Plan

Last updated: 2026-07-07.

Doc-role note (2026-07-07, resolves a two-masters ambiguity): `NORTH_STAR_ROADMAP.md` is the
master phased TO-DO plan (phases P0-P7/PF, task IDs, PART VI wave playbook); THIS file remains
canonical for the final goal and the truth/no-overclaim boundaries below. On planning/sequencing
conflicts, NORTH_STAR wins; on truth-claim boundaries, this file + `CAPABILITIES.md` win.

## Final Goal

Build the best single-camera pickleball video-to-3D analysis pipeline:

- The iPhone records a stable, landscape, full-court video and gives fast
  on-device guidance.
- The upload includes the video, capture/court sidecar, and lightweight priors.
- The server runs the accurate offline pipeline and produces a trust-banded
  scrubber bundle: video sync, court map, player/ball/contact timelines, 3D
  world, replay assets, and coaching facts.
- Every output is confidence-badged. A missing or low-confidence result is shown
  as missing/preview, never as verified truth.

This product is single-camera for v1. Multi-camera is training-time or future
scope only.

## What Exists So Far

This section lists repo reality, not promotion evidence:

- A Python offline pipeline entrypoint exists at
  `scripts/racketsport/process_video.py`. It can ingest a video, consume trusted
  calibration inputs, run or reuse stage artifacts, write `PIPELINE_SUMMARY.json`,
  and produce a replay-viewer manifest for inspectable bundles.
- Stage implementations and contracts live under `threed/racketsport/`, with
  focused CLIs in `scripts/racketsport/` and tests under `tests/racketsport/`.
- Native iOS module boundaries exist under `ios/` for capture, calibration,
  fast-tier cues, guidance, upload, and replay. These are not full device proof.
- `web/replay/` is the current review viewer for trust-banded replay bundles.
  It is a QA surface, not a production replay proof by itself.
- `docs/racketsport/` is reserved for JSON schemas/manifests. Narrative research
  or status belongs in canonical root docs if still current, or under `runs/` as
  generated evidence for a specific experiment.
- `runs/` is generated evidence. It is large by design and must not be treated as
  source code or current truth without rechecking the exact run and command.

## Current Truth

`VERIFIED=0`.

The current `scripts/racketsport/process_video.py` glue can complete scoped
accepted-clip runs and write scrubber-ready artifacts, but that is not a global
acceptance pass. Recent speed and visual runs made the bundle faster and more
inspectable, but CAL, TRK, BALL, BODY, RKT, replay, and E2E all remain below
their promotion gates.

| Area | Current status | What is true now | Promotion gate |
|---|---|---|---|
| CAL | Scaffold/preview | Manual/sidecar and metric-15pt calibration paths can feed the pipeline. The 2026-07-05 court-autofind Wave A patch is worktree-only, not mainline, and still misses the aggregate hard bar. | Held-out PCK@5px gate on reviewed owner viewpoints. |
| TRK | In progress | YOLO26m plus BoT-SORT/ReID/raw-pool tooling exists. Pre-registered gate runs still fail coverage/identity/spectator constraints. | Per-clip IDF1, zero ID switches where required, zero true spectator/background FP, coverage gate. |
| BALL | Scaffold | The default 3D ball chain landed in commit `790930ed`, but held-out quality did not promote: the product chain is cleaner and lower-FP than the standing zero-shot baseline. The wave-4 SST-3k student then beat the standing zero-shot bar on both clips in harness_v0 mode (F1 0.744/0.727, visible-hit recall 0.771; `runs/lanes/w4_ballgpu_20260707/REPORT.md`), but every card stays NON-PROMOTABLE (preprocessing-contract mismatch, small-N, 3k/12k steps) pending a wave-5 official-mode re-score. | Reviewed ball F1/contact/in-out gates, with gray-zone behavior for uncertain calls. |
| BODY | Scaffold | Fast SAM-3D-Body runtime, mesh index artifacts, speed improvements, and stance-aware visual smoothing exist for scoped clips. Candidate-label review and current visual runs are not independent GT. | Representative independent-GT world-MPJPE gate. |
| FOOT/PHYS | Internal-val done | Wolverine internal-val foot-lock reached zero slide/penetration in a scoped chain. | Strict protected-clip physics/replay verification before user-facing promotion. |
| RKT | Scaffold | Phase-1 fused paddle estimator final_v3 is render-only: useful internal-val paddle orientation/position for review, no true-reference GT promotion. | Face-angle/contact-point error against true-corner/reference GT. |
| iOS | Scoped passes | Swift modules, capture/import sidecars, upload manifest, and live-tier slices have tests. Physical capture/import/live thermal/render proof is still incomplete. | Real device capture plus live overlay and replay verification under the documented budget. |
| Replay | Scoped review pass | Web review viewer and scoped mesh-index/replay artifacts can load review bundles. Visual artifacts are viewer-consumable, not product replay proof. | Production replay asset, native/web perf, and visual QA gates. |
| E2E | SCAFFOLD/SCOPED PASS, not VERIFIED | `process_video.py` can write complete or partial bundles with fail-closed trust bands. The latest Wolverine visual run is faster, but still partial and ball/BODY gates remain scoped. | One real clip meeting component quality gates and replay SLA from a clean command. |

## 2026-07-05 Evidence Reconciliation

The following claims are evidence-linked but do not change `VERIFIED=0`:

- Speed: Wolverine E2E moved from 2141.1s to 1144.0s in the speed handoff
  (`runs/lanes/pipeline_speed_accuracy_handoff_20260705/HANDOFF.md`), then to
  702.425s after chunkfix
  (`runs/body_chunkfix_verify_20260705T204618Z/PIPELINE_SUMMARY.json`), and to
  532.252s in the latest visual run
  (`runs/visual1_wolverine_20260705T220517Z/PIPELINE_SUMMARY.json`). BODY
  remote-instrumented `stage_wall_seconds` moved from 618.536s to 473.037s
  across chunkfix and payload-collapse timing artifacts
  (`runs/body_chunkfix_verify_20260705T204618Z/source/body_stage_phase_timing.json`,
  `runs/body_payload_collapse_verify_20260705T221027Z/source/body_stage_phase_timing.json`).
  The payload-collapse live report records this as a real speed win, but also
  records accuracy-invariant divergence caused by an entangled concurrent
  `worldhmr.py` change; do not claim a clean isolation verdict from it
  (`runs/lanes/body_payload_collapse_livecheck_20260705/REPORT.md`).
- Visual BODY review: stance-aware smoothing and placement redistribution are
  live in the visual artifact. Reset count moved from 14 to 2 and worst
  root-step p95 moved from 0.267m/frame to 0.100m/frame in the before/after
  visual-quality artifacts
  (`runs/lanes/visual_polish_20260705/lane_VPA2_resets_jitter/before_speed1_wolverine_source/visual_quality.json`,
  `runs/lanes/visual_polish_20260705/after_visual1/visual_quality.json`).
  The same lane notes that ball runtime was absent in that run, so contact-dense
  mesh scheduling fell back to uniform coverage
  (`runs/lanes/visual_polish_20260705/STATUS.md`).
- Ball: the default 3D chain landed in `790930ed`, but the held-out product
  chain scored 0.6969 F1 versus the standing zero-shot WASB-tennis 0.7248 bar;
  it improved precision, hidden-FP, tail error, and teleports while losing
  recall. Evidence:
  `runs/lanes/ball_heldout_chain_run_20260704/scoring/outdoor_webcam_iynbd_1500_long_high_baseline_benchmark.json`,
  `runs/lanes/ball_tracking_long_run_STATUS.md`, and
  `runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md`.
- Racket: phase-1 fused paddle estimator final_v3 is render-only. Internal-val
  scoring is Wolverine IoU 0.2356 and Burlington IoU 0.3424 with zero
  undeclared >0.35m teleports after final_v3; five declared switch jumps remain.
  Evidence:
  `runs/lanes/racket_6dof_20260705/i1_fused_estimator/acceptance_record_v2.json`
  and `runs/lanes/racket_6dof_20260705/STATUS.md`.
- Court auto-find: Wave A evidence is a handoff patch on worktree branch
  `worktree-court-autofind-20260705`, not a mainline capability. Outdoor no-tap
  median is 4.4px, but aggregate mean floor median is 213.3px versus the 200px
  hard bar and the prior 289.5px baseline. Evidence:
  `runs/lanes/court_autofind_20260705/handoff/cal_geo_r2_report.md` and
  `runs/lanes/court_autofind_20260705/handoff/court_autofind_wave_a.patch`.

## Automatic Court Finding Research Snapshot

Generated evidence and the detailed research handoff live under
`runs/court_finding_technology_benchmark_20260703/`. Do not move this narrative
under `docs/racketsport/`; that directory is reserved for JSON schemas and
manifests.

Current five-sample court-finding benchmark:

- CLI:
  `python3 scripts/racketsport/evaluate_court_finding_technologies.py --eval-root eval_clips/ball --out-dir runs/court_finding_technology_benchmark_20260703 ...`
- Samples: the four full 15-point CVAT/eval clips plus owner
  `owner_IMG_1605_8a193402780b` partial visible labels.
- Artifact:
  `runs/court_finding_technology_benchmark_20260703/court_finding_technology_benchmark.json`.
- Status remains `ran_not_verified`, `verified=false`,
  `not_cal3_verified=true`.

Measured state as of 2026-07-03:

| Candidate | Mean floor median px | Mean floor p95 px | Notes |
|---|---:|---:|---|
| `hough_or_refined_regulation_line_selector` | 289.5 | 551.0 | Best current deployable median prototype; uses guarded line-refined regulation on Indoor and `IMG_1605`; still mandatory-review only. |
| `hough_or_regulation_line_selector` | 296.4 | 547.0 | Previous best deployable median prototype after projected-pixel/color scoring and geometry guard; still mandatory-review only. |
| `hough_regulation_temporal_balanced_selector` | 296.4 | 547.0 | Matches the simple selector unless temporal evidence is a clear internal-score win; still mandatory-review only. |
| `hough_or_regulation_distance_mask_selector` | 298.4 | 531.8 | Best current median/p95 compromise; uses distance-mask regulation on Indoor only. |
| `hough_regulation_temporal_line_selector` | 310.3 | 581.3 | Naive temporal selector is no longer best because it can worsen Indoor p95. |
| `hough_regulation_temporal_persistent_tail_selector` | 330.4 | 518.3 | Best current deployable p95/tail-risk tradeoff; still mandatory-review only. |
| `opencv_hough_lsd_regulation_line_refined` | 331.7 | 652.7 | Guarded point-and-line homography refinement improves median slightly, but worsens p95 slightly; not best standalone. |
| `opencv_hough_lsd_regulation` | 338.7 | 648.7 | Pixel/color scoring improved median strongly, but Burlington remains tail-risky. |
| `opencv_hough_lsd_skimage_regulation` | 452.3 | 880.3 | Seeded skimage probabilistic Hough merged into OpenCV Hough+LSD was measured and is worse than current OpenCV regulation. |
| `reviewed_oracle_hough_regulation_temporal` | 243.9 | 826.2 | Reviewed-label oracle; benchmark-only, not deployable. |

Conclusion: net evidence is useful as a prior and 3D validator, but the needed
system is a net-anchored line-to-regulation optimizer with tennis-template
negative scoring, color/mask evidence, temporal persistence, robust nonlinear
refinement, and p95/worst-corner gates. The median and tail-risk selectors are
both better than the previous baseline in different ways, but neither is close
to automatic promotion.

Tennis-service template competition is now recorded in regulation proposals and
covered by a synthetic spacing test. It did not move the current five-sample
metrics because the selected hypotheses are not tennis-service-template-like by
cross-line spacing; broader tennis rejection still needs sideline-width,
service-line, overlong-line, and color/mask competition.

Projected regulation-line pixel support, distance-transform line-mask support,
local line-color/layer consistency, shadow-normalized Hough, guarded scipy
point-and-line homography refinement, and selector geometry guards are now
implemented in the benchmark. Current evidence: pixel/color scoring improved
single-frame regulation median from the previous `459.2 px` to `338.7 px`;
guarded line refinement improves the best deployable median selector to
`289.5 px` by upgrading Indoor and `IMG_1605`, but standalone refinement worsens
mean p95 slightly (`652.7 px` vs `648.7 px`); guarded distance-mask selection
improves p95 versus the older median selector (`531.8 px` vs `547.0 px`) with a
small median cost; the geometry guards keep Burlington on `hough_keypoints` and
block Wolverine's tiny distance-mask court; strict HSV paint regulation failed
to build proposals on all five samples; shadow-normalized Hough underperformed
plain Hough line support (`0.7167` vs `0.8083`).
An unguarded nonlinear refinement probe overfit the Outdoor sample, moving its
median from `12.7 px` to roughly `333 px` despite a better assigned-line
residual, so future optimizers must include projected-pixel/mask support,
point-drift, p95, and tennis-negative self-verification gates.
OpenCV contrib Fast Line Detector and skimage probabilistic Hough are now
benchmarked and runnable. Fast Line Detector scored `0.8417` floor-line support,
seeded skimage scored `0.8167`, and the merged OpenCV+LSD+skimage regulation
solver stayed worse than current OpenCV regulation (`452.3 px` mean floor median
vs `338.7 px`). ELSED is wired as a fail-closed optional adapter, but local
`pyelsed` installation is blocked by native OpenCV development package
availability after the upstream CMake compatibility issue is patched.

Additional 2026-07-03 research points to the next practical CAL lane: keep the
net as an anchor/validator, but add swappable deep/classical line detectors
(`DeepLSD`, `ScaleLSD`, and dependency-isolated `ELSED`), temporal line identity
(`SOLD2`/`GlueStick` or classical persistence), SAM 2 or learned masks as evidence, and a robust
point-and-line optimizer with explicit pickleball-vs-tennis template margins.
These are research/implementation candidates only; they still must beat the
five-sample benchmark and remain `verified=false` until reviewed gates pass.

The overlapping or multipurpose court paint path is evidence-only research, not
a promoted calibration route. Current code has HSV paint masking, near-side net
crop anchoring, clustered Hough boundary extraction, LabelMe point loading,
ResNet50 keypoint-regression scaffolding, and LM homography residual reporting.
The reviewed-label report CLI is
`scripts/racketsport/evaluate_overlapping_court_calibration.py`; its outputs
must stay `verified=false` / `not_cal3_verified=true` until reviewed held-out
no-tap gates pass. If the paint-color assumption is false on a real clip, the
correct outcome is low or zero candidate support, not silent promotion.
Current reviewed-label evidence from `runs/overlapping_court_calibration_20260703/`:
LM-optimized mean residual: 0.414584 ft over 4 full clips, with only 1 / 4 clips
passing the 0.2 ft mean target. HSV/Hough reviewed-line support is weak on the
current labels: `opencv_hsv_paint_hough`: 0.0000 and
`opencv_hsv_paint_net_crop_hough`: 0.0250, versus baseline `opencv_hough`:
0.8083. That means the current reviewed clips do not support silent promotion of
strict colored-line masking.
The strongest new diagnostic combines full-intrinsics metric-plane fitting with
strict top-residual line-intersection replacements: 0.193404 ft mean residual on
the temporary override observations, but 0.408027 ft when scored against the
original reviewed labels. Keep it as a review/outlier-localization result only;
the safe selected camera remains the fixed-center metric-plane fit at 0.332284 ft.
A less label-selected all-strict endpoint line-intersection policy was measured
too: 30 endpoint replacements, 0.230184 ft on temporary override observations,
and 0.655275 ft against original reviewed labels. That misses the 0.2 ft target
and shows unfiltered endpoint line intersections are too noisy for promotion.
A fixed line-quality sweep now gates endpoint intersections by angle agreement,
perpendicular distance, segment overlap, and optional proximity to the current
full-intrinsics model projection. The best profile,
`tight_overlap35_dist12_angle8_model24`, improves the all-strict endpoint
diagnostic to 0.182784 ft with 25 replacements, but it is still diagnostic: the
worst clip is 0.236466 ft, the fit scores 0.494922 ft against original reviewed
labels, and the safe selected camera remains 0.332284 ft. A new model-projected
line-observation lane reproduces the 0.182784 ft temporary result while
reporting `uses_reviewed_line_positions_for_matching=false`, so reviewed line
matching is no longer required for that diagnostic. Treat it as evidence for
model-proximity/template competition, not a calibration promotion.

## Architecture

**L0/L1 live advisory tier:** AVFoundation capture, ARKit/manual calibration
seed, capture-quality guidance, lightweight person/ball/pose cues, court map,
one priority cue, upload priors, and post-stop replay summaries. This tier is
fast, conservative, and uncertainty-banded.

**L2/L3 server tiers:** L2 is the trust-banded fast verdict path that skips BODY;
L3 is the deep-world authority with calibration refinement, deep tracking,
ball/event processing, Fast SAM-3D-Body mesh, grounding, foot-lock/physics,
paddle 6DoF when available, metrics, replay bake, and coaching copy.
`CAPABILITIES.md` owns the canonical four-tier split.

**GPU reset state:** the July 2026 A100 spot runtime is reset-pending during
winddown. Do not treat any named VM as currently available without a fresh
connectivity, disk, and GPU-lock check.

**RTMW retirement:** RTMW/RTMW3D/RTMPose are retired from the pipeline. BODY is
SAM-3D-Body only: Fast SAM-3D-Body is the offline skeleton/mesh source because
the prior RTMW family was less accurate on visible pickleball joints, and the
current optimized SAM-3D-Body path reached equal-or-better speed while producing
the mesh/joint artifacts the replay stack needs.

`CAPABILITIES.md` owns the exact tier split. `RUNBOOK.md` owns the runnable
pipeline command.

## Non-Negotiable Rules

- Do not train on protected eval clips unless the code path explicitly marks an
  internal-val diagnostic. Outdoor and Indoor are strict holdout clips.
- Do not call a stage `VERIFIED` from smoke tests, schema validation, copied
  fixtures, internal-val-only evidence, browser loads, or a partial pipeline run.
- Do not reuse killed levers without new evidence. In particular, avoid
  re-running CVAT-only BALL fine-tunes, BALL local-search postprocess, TRK
  association-only sweeps on the exhausted lever, BODY candidate-label promotion,
  and paddle rectangle-to-6DoF promotion.
- Keep all status wording tied to a specific command, run path, test result, or
  device/runtime observation.

## Next Gates

1. Make CAL fail-closed and usable: tap-assisted/metric seed stays v1, no-tap
   remains unverified until reviewed gates pass.
2. Improve TRK with real detector/data leverage, not another exhausted association
   sweep.
3. Improve BALL with reviewed data and a model-side candidate that beats the
   current confidence-gated baseline without hidden-FP or recall regressions.
4. Promote BODY only from independent GT, not candidate labels.
5. Complete iOS physical-device capture/import/live-overlay/replay proof.
6. Run a clean `process_video.py` reproduction after component gates improve.

## Documentation Policy

Canonical narrative docs are intentionally small:

- `README.md`
- `AGENTS.md`
- `MASTER_PLAN.md`
- `RUNBOOK.md`
- `CAPABILITIES.md`
- `BUILD_CHECKLIST.md`
- `TECH_STACK.md`
- `BALL_TRACKING_PIPELINE.md`
- `TIER_MAP.md` as a derived quick reference only

Generated evidence belongs under `runs/`. JSON contracts and schemas belong
under `docs/racketsport/`. Do not add new long-lived narrative docs unless one
of the canonical docs cannot hold the information cleanly.
