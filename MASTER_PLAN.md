# Pickleball Master Plan

Last updated: 2026-07-03.

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
acceptance pass. The scoped Wolverine run recorded in prior evidence completed
the intended stage chain and produced a replay manifest, but CAL remained
metric-preview, TRK remained do-not-promote, BALL remained low-confidence, and
BODY still lacked the representative world-MPJPE gate.

| Area | Current status | What is true now | Promotion gate |
|---|---|---|---|
| CAL | Scaffold/preview | Manual/sidecar and metric-15pt calibration paths can feed the pipeline. No no-tap automatic court solver has passed. | Held-out PCK@5px gate on reviewed owner viewpoints. |
| TRK | In progress | YOLO26m plus BoT-SORT/ReID/raw-pool tooling exists. Pre-registered gate runs still fail coverage/identity/spectator constraints. | Per-clip IDF1, zero ID switches where required, zero true spectator/background FP, coverage gate. |
| BALL | Scaffold | WASB/TrackNet tooling, reviewed labels, bounce/in-out utilities, audio onset, and confidence-gated display paths exist. M0-M8 remains unpromoted. | Reviewed ball F1/contact/in-out gates, with gray-zone behavior for uncertain calls. |
| BODY | Scaffold | Fast SAM-3D-Body runtime and structural BODY artifacts exist for scoped clips. Candidate-label review cannot become independent GT. | Representative independent-GT world-MPJPE gate. |
| FOOT/PHYS | Internal-val done | Wolverine internal-val foot-lock reached zero slide/penetration in a scoped chain. | Strict protected-clip physics/replay verification before user-facing promotion. |
| RKT | Scaffold | Paddle boxes, masks, and review candidates exist. True paddle-face corner/reference GT is still missing. | Face-angle/contact-point error against true-corner/reference GT. |
| iOS | Scoped passes | Swift modules, capture/import sidecars, upload manifest, and live-tier slices have tests. Physical capture/import/live thermal/render proof is still incomplete. | Real device capture plus live overlay and replay verification under the documented budget. |
| Replay | Scoped review pass | Web review viewer and scoped GLB/USDZ/replay artifacts can load and display trust-banded data. | Production replay asset, native/web perf, and visual QA gates. |
| E2E | SCAFFOLD/SCOPED PASS, not VERIFIED | `process_video.py` can write a complete/partial bundle with fail-closed trust bands. | One real clip meeting component quality gates and replay SLA from a clean command. |

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
| `hough_or_regulation_line_selector` | 296.4 | 547.0 | Best current deployable median prototype after projected-pixel/color scoring and geometry guard; still mandatory-review only. |
| `hough_regulation_temporal_balanced_selector` | 296.4 | 547.0 | Matches the simple selector unless temporal evidence is a clear internal-score win; still mandatory-review only. |
| `hough_regulation_temporal_line_selector` | 310.3 | 581.3 | Naive temporal selector is no longer best because it can worsen Indoor p95. |
| `hough_regulation_temporal_persistent_tail_selector` | 330.4 | 518.3 | Best current deployable p95/tail-risk tradeoff; still mandatory-review only. |
| `opencv_hough_lsd_regulation` | 338.7 | 648.7 | Pixel/color scoring improved median strongly, but Burlington remains tail-risky. |
| `reviewed_oracle_hough_regulation_temporal` | 273.9 | 721.6 | Reviewed-label oracle; benchmark-only, not deployable. |

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

Projected regulation-line pixel support, local line-color/layer consistency,
shadow-normalized Hough, and selector geometry guards are now implemented in the
benchmark. Current evidence: pixel/color scoring improved single-frame
regulation median from the previous `459.2 px` to `338.7 px`; the geometry guard
keeps Burlington on `hough_keypoints` because the visually supported regulation
proposal is floor-collapsed; strict HSV paint regulation failed to build
proposals on all five samples; shadow-normalized Hough underperformed plain
Hough line support (`0.7167` vs `0.8083`).

Additional 2026-07-03 research points to the next practical CAL lane: keep the
net as an anchor/validator, but add swappable deep/classical line detectors
(`ELSED`, `DeepLSD`, `ScaleLSD`), temporal line identity (`SOLD2`/`GlueStick` or
classical persistence), SAM 2 or learned masks as evidence, and a robust
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

## Architecture

**On-device live tier:** AVFoundation capture, ARKit/manual calibration seed,
capture-quality guidance, lightweight person/ball/pose cues, court map, one
priority cue, upload priors. This tier is fast and conservative.

**Server offline tier:** calibration refinement, deep tracking, ball/event
processing, Fast SAM-3D-Body mesh, grounding, foot-lock/physics, paddle 6DoF
when available, metrics, replay bake, and coaching copy. This tier is the
accuracy authority.

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
