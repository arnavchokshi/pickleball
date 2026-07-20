# Court line hardening results — 2026-07-20

`VERIFIED=0` · preview/research only · no default change · no runner wiring · no commit

## Verdict

The preregistered product acceptance target **did not pass**. The final
default-off candidate did not improve any frozen venue at the scorer's
published precision, and it did not improve median or p90 on either frozen
CPM-v2 diagnostic clip. No venue materially regressed because every scored
output remained the unchanged seed.

No lever is selected for promotion or runtime wiring. The implementation is
retained only as an unwired, default-off preview/evidence surface. An honest
negative is the result of this lane.

## Frozen source and evidence

- `threed/racketsport/court_line_robustness.py`:
  `d1e51b708624e2885e5ebfcb6b15411ef049e29d20f2d4ebbc1cb7a57d4a5167`
- `threed/racketsport/court_line_keypoints.py`:
  `1ac1cccfca8b5ae10aee2c7e658e108f845d856f5ab3d9e979228079b15c9081`
- Final ultra review: no remaining Critical/High code blocker at those hashes.
- Five-venue artifact:
  `runs/lanes/court_line_hardening_20260720/five_venue_enabled_evaluation.json`
  (`sha256=208b62a294d3d7d38e6b10c8931d23cf2f4142f29d9a38357a2a368be1127536`).
- Corrected CPM-v2 diagnostic:
  `runs/lanes/court_line_hardening_20260720/corrected_two_clip_diagnostic/evaluation.json`
  (canonical `sha256=316c5907580fa4a41dd1f7226922de013dd37e95a5278ff7a74e36e1eb7ce202`;
  file `sha256=8c1c3f95b2f32684af6423f8fc05fe17b5ff889c5b3f3ed20610a832c229d7a1`).
- The earlier apparent gains are rejected. They used reviewed scorer
  correspondences in the candidate fit and remain quarantined under
  `rejected_candidate_evaluation_reviewed_point_leakage/`.

The five-venue evaluator built its frame input from `git archive HEAD
eval_clips/ball`, locked and hashed automatic Hough proposals before opening
reviewed targets, and scored every abstention as the unchanged seed. The two
diagnostic clips deliberately retain their reviewed 15-point calibrations only
as scorer inputs; fail-closed provenance prevents those points from entering a
candidate fit.

## Reproduced baselines

### Frozen reviewed five-venue harness

The exact frozen metric is visible floor-keypoint median/p95 in native pixels.

| Venue | Baseline median | Baseline p95 |
|---|---:|---:|
| Burlington | 467.2801 | 649.2293 |
| Indoor doubles | 257.6769 | 669.4432 |
| Outdoor webcam | 11.8818 | 21.1551 |
| Owner IMG_1605 | 798.3121 | 1247.7156 |
| Wolverine | 672.3342 | 1230.4436 |
| Aggregate mean / median-of-medians | 441.4970 / 467.2801 | mean 763.5974; max 1247.7156 |

### Frozen CPM-v2 diagnostic

This diagnostic supplies the requested p90 check and remains non-promotional.

| Clip | Baseline median | Baseline p90 | After median | After p90 | Delta |
|---|---:|---:|---:|---:|---:|
| Wolverine | 3.005485 | 6.217355 | 3.005485 | 6.217355 | 0 / 0 |
| Burlington | 3.804831 | 8.856592 | 3.804831 | 8.856592 | 0 / 0 |

The frozen evidence hashes, payloads, coverage, and residual summaries match
the banked baseline exactly.

## Final five-venue result

The sub-picopixel values below are serialization/adapter epsilon. Every row is
exactly unchanged at the frozen report precision of four decimals.

| Venue | After median | After p95 | Median delta | p95 delta | Pool / refinement |
|---|---:|---:|---:|---:|---|
| Burlington | 467.2801 | 649.2293 | -2.10e-11 | +3.87e-12 | pool abstained / seed |
| Indoor doubles | 257.6769 | 669.4432 | -1.14e-13 | -3.55e-11 | pool abstained / seed |
| Outdoor webcam | 11.8818 | 21.1551 | -1.60e-13 | +3.13e-13 | pool accepted / rejected: held-out left sideline |
| Owner IMG_1605 | 798.3121 | 1247.7156 | -5.68e-13 | +1.14e-12 | one frame; pool abstained / seed |
| Wolverine | 672.3342 | 1230.4436 | 0 | -1.36e-12 | pool abstained / seed |

Summary: pools accepted `1/5`; refinements accepted `0/5`; improvements at
four decimals `0/5`; material regressions `0/5`.

## Lever-by-lever decision

These were run sequentially, so a zero in this table means the public scored
output remained the seed; it is not presented as an isolated causal ablation.

| Lever | Burlington | Indoor | Outdoor | IMG_1605 | Wolverine | Decision |
|---|---|---|---|---|---|---|
| Regulation ROI + lookalike rejection | 0 at 4dp | 0 at 4dp | 0 at 4dp | 0 at 4dp | 0 at 4dp | Do not promote. Adversarial unit cases pass, but the frozen harness did not improve. |
| Static cross-frame robust-median pooling | 0; abstain | 0; abstain | 0; pool accepted | 0; only one frame | 0; abstain | Do not promote. Deterministic/provenanced, but no scored win. |
| Line-over-point fit (`0.60 / 0.40`) | not reached | not reached | 0; optimizer rejected on held-out left sideline | not reached | not reached | Do not select a new weight. The line-only `1.0 / 0.0` arm is explicitly diagnostic and non-promotional. |
| Shadow removal | not run | not run | not run | not run | not run | Correctly not implemented: no frozen shadow stratum was measured as the causal failure, and repainting could erase paint. |

The additive legacy/Hough and hybrid candidate providers are non-default.
Hybrid center fusion was not measured and is not selected.

## Integrity and determinism

- Feature off: same object and canonical bytes on `5/5` primary venues and
  `2/2` CPM-v2 diagnostic clips.
- Deterministic rebuild: byte-identical on `4/4` eligible multiframe primary
  venues and `2/2` diagnostic clips. IMG_1605 has only one tracked frame.
- Raw per-frame evidence is immutable; pooled evidence is a separate artifact
  with contributing/rejected frame indexes and hashes.
- Automatic point evidence is bound by authority, source, artifact SHA, exact
  correspondences SHA, canonical world-point values/order, seed hash, template
  projection hash, image size, coordinate space, distortion state, and config
  hash.
- Reviewed/manual/scorer points fail closed before optimizer invocation.
- Only the existing guarded point/line optimizer is used. A damped
  optimizer-selected step is not treated as a preregistered win.
- Accepted homography-only previews invalidate stale pose, contract,
  provenance, trust, and all-15 reprojection fields; they emit explicitly
  floor-only diagnostics.

## Verification

- Focused blast-radius command:

  ```bash
  MPLBACKEND=Agg .venv/bin/python -m pytest -q \
    tests/racketsport/test_court_line_hardening.py \
    tests/racketsport/test_court_line_keypoints.py \
    tests/racketsport/test_court_paint_centerline.py \
    tests/racketsport/test_court_calibration.py \
    tests/racketsport/test_court_calibration_distortion.py \
    tests/racketsport/test_court_precision_metrics.py \
    tests/racketsport/test_court_precision_harness_cli.py
  ```

  Result: `84 passed`.

- Full wide command:

  ```bash
  MPLBACKEND=Agg .venv/bin/python -m pytest -q tests/racketsport
  ```

  Result: `3913 passed, 26 failed, 25 skipped` in `47m40s`.

  None of the 26 failures exercises the new hardening module or its focused
  tests:

  - 18 are pre-existing workspace/data-fixture drift around IMG_1605 and eval
    discovery: the live directory contains three frames where tests freeze one;
    the local partial-label progress exposes 15 points where tests expect 14;
    an extra eval clip changes a five-sample expectation to six; downstream
    selectors consequently score four clips instead of five.
  - 8 are environment failures because this sandbox denies TCP or Unix-domain
    socket binds: three court-keypoint review-server tests, three generic
    review-server tests, and two persistent BODY-worker tests.

  A representative `git archive HEAD` snapshot rerun also failed before this
  lane's files were overlaid: socket tests reproduced `PermissionError:
  Operation not permitted`; IMG_1605 tests could not find their untracked
  review-progress prerequisite. Because not every raw-workspace failure was
  reproduced with the exact same reason at HEAD, this report does **not** set
  `failures_all_preexisting=true`. The scoped 84-test court blast radius is
  green.

## Runner wiring hunk — intentionally not applied

There is no evidence-backed reason to wire this candidate into the sole
pipeline entrypoint. If a future frozen run produces a real win, the
integration owner could add a default-off preview artifact seam shaped like
the following. It must not overwrite `court_calibration.json` or change its
authority:

```diff
diff --git a/scripts/racketsport/process_video.py b/scripts/racketsport/process_video.py
--- a/scripts/racketsport/process_video.py
+++ b/scripts/racketsport/process_video.py
@@
+from threed.racketsport.court_line_robustness import (
+    CourtLineHardeningConfig,
+    CourtLineHardeningResult,
+    maybe_apply_court_line_hardening,
+)
@@ def _stage_calibration(self) -> StageOutcome:
         payload = _read_json(target)
+        line_config = CourtLineHardeningConfig(enabled=False)
+        line_preview = maybe_apply_court_line_hardening(
+            _decode_preregistered_static_court_frames(opts.video),
+            payload,
+            config=line_config,
+        )
+        if isinstance(line_preview, CourtLineHardeningResult):
+            _write_json(
+                self.clip_dir / "court_line_hardening_preview.json",
+                line_preview.as_dict(),
+            )
         self._set_court_trust_band(payload, target)
```

This hunk is illustrative and **not applied**. A real integration would also
need an explicit CLI/config surface, static-camera/motion eligibility,
preregistered frame sampling, schema validation, and focused orchestration
tests.

## Remaining gap

The detector keeps one locally ranked paint band per profile. Joint k-best
assignment can recover among candidate-level alternatives, but it cannot
recover a true paint band already discarded inside local profile detection.
The default hybrid path also keeps the longest candidate geometry instead of
measuring/fusing subpixel centers. More importantly, the frozen venue errors
remain enormous on four of five clips, so this preview does not close the
product court-calibration gap.

The next bounded experiment, if court work continues, is a multi-band
seed-guided evidence artifact scored on the same frozen venues, with no solver
or weight change. Until that beats the baseline, `VERIFIED=0` remains binding.
