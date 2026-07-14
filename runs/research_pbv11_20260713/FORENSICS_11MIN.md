# pb.vision 11-minute export deep forensics

Date: 2026-07-13  
Status: `VERIFIED=0`  
Policy: pb.vision is a competitor-reference diagnostic, never ground truth or training data.  
Best-stack delta: **(c) none**.

## Bottom line

The 11-minute export preserves the prior method-family diagnosis but refutes
several literal universal claims. It exports one static camera solve, 42 CV
rally windows, 41 downstream product rallies, 7,808 selected 3D samples over
10,322 rally frames (75.64% coverage), strong event-boundary velocity breaks,
rare explicit interpolation, and a broadly useful radius/depth cue.

The important scale corrections are:

- only 161/170 selected bounces are at the radius plane; all 9 exceptions are
  explicitly `out_of_sequence`, while all 154 in-sequence bounces are pinned;
- in-sequence bounce anchors reproject to numerical zero, but shot anchors are
  not universally hard ray constraints: 245/260 in-sequence shots are within
  `1e-6 px`, with 15 nonzero first-shot/serve anchors;
- piecewise gravity+drag re-integration passes 392/422 evaluable segments
  (92.89%), not 100%; excluding segments containing an out-of-sequence event
  yields 364/388 (93.81%);
- the radius cue is directionally present in all 41 eligible rallies and has
  Spearman >=0.5 in all 41, but pooled linear strength falls from the prior
  one-rally `r/R2=0.850/0.722` to `0.535/0.287`; it is useful, not universal as
  one global linear calibration; and
- longer arcs still do not identify meaningful drag: only 2/130 meet the local
  diagnostic and both fail the declared physics-plausibility rule, leaving
  0/130 credible pass-compatible identifications.

The convergent engineering ruling is now stronger: accepted event evidence is
the first bottleneck. TT3D was rejected, recovery made no pb-rally coverage
gain, and whole-rally DP remains killed. The source video is now available;
freeze a same-clip baseline, improve audio/blur anchors, then ablate radius plus a new
reviewed-anchor robust fit before retrying recovery.

## Inputs and reproducibility

| Input | Version | Bytes | SHA-256 |
|---|---:|---:|---|
| `data/pbvision_11min_20260713/cv_export.json` | 2.1.0 | 12,454,922 | `4ccd08fd68f205b6cb0b2aa6a51aacd5ae58e12f0aabaf8687d1e453077a2ffc` |
| `data/pbvision_11min_20260713/insights.json` | 4.10.0 | 309,714 | `fbe3e16299dc96dbe70528a869ad4b96319d2961b47259207b686e4094ff1cea` |
| `data/pbvision_11min_20260713/stats.json` | 2.2.0 | 51,915 | `09a6da41145e9b69884f80ef2ac47389c351b6cd7d3b0361a08e6a51601f6af4` |
| `data/pbvision_11min_20260713/source_video.mp4` | H.264 1280x720 30fps, 697.4s | 119,911,668 | `272a2132ce7c72ea31fe6351c9ea05ac3016bbbfed0a5801d9c3a973ec628383` |
| `data/pbvision_11min_20260713/video_provenance.json` | owner video-hunt evidence | 3,390 | `2fcad2c2faa94978f882ceb2b20f23d32074f222c5aa95fb5fecabb60020c699` |
| `runs/lanes/ball_recovery_20260712/report.json` | lane report | - | `1bac4e7d28680949684879cb5016586f04badd0a94b836f53b7cb06959fc90aa` |
| `runs/lanes/tt3d_integrate_20260712/report.json` | lane report | - | `1cbb28e6458ae02376b5c145ac09cd4c256f3e79fa95a31d654d25e4ffbdb` |

The source video and provenance appeared after this lane began. They were kept
read-only and used to replace the initial image-size assumption with the
verified 1280x720 grid. The camera projection still uses the prior inferred
`zyx(-yaw,pitch,-roll) -> [y,-z,x]` convention. Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  runs/lanes/pbv11_forensics_20260713/inventory_pbvision_export.py \
  data/pbvision_11min_20260713/cv_export.json \
  data/pbvision_11min_20260713/insights.json \
  data/pbvision_11min_20260713/stats.json \
  --output runs/lanes/pbv11_forensics_20260713/schema_inventory.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  runs/lanes/pbv11_forensics_20260713/compare_vs_pbvision.py \
  --cv-export data/pbvision_11min_20260713/cv_export.json \
  --insights data/pbvision_11min_20260713/insights.json \
  --stats data/pbvision_11min_20260713/stats.json \
  --ball-recovery-report runs/lanes/ball_recovery_20260712/report.json \
  --tt3d-report runs/lanes/tt3d_integrate_20260712/report.json \
  --source-video data/pbvision_11min_20260713/source_video.mp4 \
  --video-provenance data/pbvision_11min_20260713/video_provenance.json \
  --output runs/lanes/pbv11_forensics_20260713/forensics_metrics.json
```

Every detailed segment, gap, rally, radius, line-call, product-surface, and
ranking row is in `forensics_metrics.json`.

## 1. Structure at scale

### Aggregate structure

| Metric | Result |
|---|---:|
| Camera FPS / full camera span | 30 / frames 0..20,921 (20,922 frames; 697.4 s) |
| CV rally windows | 42 |
| Insights product rallies | 41 |
| CV-to-insights start-time matches | 41/41 within 1 ms |
| Unmatched CV rally | index 35, frames 17,522..17,635, 114 frames, zero selected 3D |
| Rally frames | 10,322 |
| Selected 3D | 7,808 (75.644%) |
| Selected types | 7,318 ball; 292 shot; 170 bounce; 28 net |
| Insights shots | 256; 36 fewer than typed CV shot anchors |
| Explicit interpolated | 95/7,808 (1.217%) |
| Rallies with any emission | 41/42 |
| Emitting-rally coverage range | 48.31%..92.15% |

The 36-anchor CV/insights shot difference is product filtering, not evidence
that either count is ground truth. The one zero-emission CV rally is absent
from insights.

### Missing-window policy

| Omitted-window position | Runs | Frames/run min / p50 / p95 / max | Seconds/run p50 / p95 | Total frames |
|---|---:|---:|---:|---:|
| Leading | 41 | 9 / 27 / 30 / 30 | 0.900 / 1.000 | 1,091 |
| Internal | 23 across 14 rallies | 1 / 7 / 22.4 / 24 | 0.233 / 0.747 | 211 |
| Trailing | 41 | 1 / 32 / 37 / 39 | 1.067 / 1.233 | 1,098 |
| Whole rally | 1 | 114 / 114 / 114 / 114 | 3.800 / 3.800 | 114 |
| **All omitted** | **106 runs** | - | - | **2,514** |

All 41 emitting rallies trim both the start and end. Leading trim is tightly
clustered around 0.9 s; trailing trim is typically about 1.1 s. Internal gaps
are less common and shorter. This is strong evidence of a deliberate emission
window policy, not proof of the internal confidence rule.

### Per-rally structure

`L/I/T` means leading-trim frames / internal-gap count / trailing-trim frames.

| CV rally | Start | Frames | 3D | Cov % | Shot/Bounce/Net | Interp | L/I/T | Insights shots |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9 | 120 | 58 | 48.33 | 2/1/1 | 0 | 28/0/34 | 2 |
| 1 | 467 | 84 | 50 | 59.52 | 2/1/0 | 6 | 27/0/7 | 1 |
| 2 | 757 | 169 | 113 | 66.86 | 3/2/1 | 2 | 28/1/15 | 3 |
| 3 | 1,113 | 171 | 122 | 71.35 | 3/4/1 | 3 | 29/0/20 | 3 |
| 4 | 1,503 | 288 | 222 | 77.08 | 9/6/0 | 0 | 28/2/21 | 8 |
| 5 | 2,052 | 186 | 125 | 67.20 | 4/2/1 | 0 | 27/0/34 | 4 |
| 6 | 2,490 | 163 | 105 | 64.42 | 3/2/1 | 0 | 28/0/30 | 3 |
| 7 | 2,869 | 362 | 304 | 83.98 | 12/7/0 | 3 | 26/1/30 | 11 |
| 8 | 3,486 | 320 | 261 | 81.56 | 11/4/1 | 2 | 28/0/31 | 10 |
| 9 | 4,067 | 196 | 139 | 70.92 | 6/3/1 | 1 | 27/0/30 | 5 |
| 10 | 4,554 | 280 | 228 | 81.43 | 7/4/0 | 1 | 26/1/9 | 7 |
| 11 | 5,045 | 167 | 106 | 63.47 | 3/2/1 | 0 | 27/0/34 | 3 |
| 12 | 5,644 | 225 | 164 | 72.89 | 5/3/1 | 0 | 28/0/33 | 5 |
| 13 | 6,081 | 156 | 94 | 60.26 | 3/2/1 | 0 | 27/0/35 | 3 |
| 14 | 6,445 | 339 | 279 | 82.30 | 12/3/1 | 1 | 26/0/34 | 12 |
| 15 | 7,072 | 318 | 261 | 82.08 | 12/3/1 | 1 | 26/0/31 | 12 |
| 16 | 7,722 | 124 | 66 | 53.23 | 3/4/0 | 3 | 28/2/2 | 1 |
| 17 | 7,886 | 212 | 147 | 69.34 | 6/3/1 | 0 | 28/0/37 | 6 |
| 18 | 8,358 | 120 | 59 | 49.17 | 2/1/1 | 1 | 30/0/31 | 2 |
| 19 | 8,568 | 170 | 110 | 64.71 | 4/2/1 | 0 | 13/1/35 | 3 |
| 20 | 8,966 | 196 | 141 | 71.94 | 5/4/0 | 0 | 30/2/9 | 4 |
| 21 | 9,566 | 518 | 440 | 84.94 | 18/10/1 | 12 | 28/4/31 | 15 |
| 22 | 10,315 | 209 | 150 | 71.77 | 6/3/1 | 3 | 27/0/32 | 5 |
| 23 | 11,065 | 162 | 102 | 62.96 | 3/2/1 | 0 | 27/0/33 | 3 |
| 24 | 11,458 | 292 | 233 | 79.79 | 9/6/0 | 1 | 27/0/32 | 8 |
| 25 | 12,164 | 152 | 87 | 57.24 | 3/2/1 | 0 | 29/0/36 | 3 |
| 26 | 12,558 | 503 | 448 | 89.07 | 20/13/0 | 1 | 28/3/1 | 13 |
| 27 | 13,519 | 260 | 209 | 80.38 | 5/4/1 | 5 | 26/1/12 | 5 |
| 28 | 14,017 | 118 | 57 | 48.31 | 2/1/1 | 1 | 27/0/34 | 2 |
| 29 | 14,366 | 446 | 411 | 92.15 | 18/10/0 | 21 | 27/0/8 | 13 |
| 30 | 15,022 | 167 | 120 | 71.86 | 5/2/1 | 1 | 9/1/32 | 2 |
| 31 | 15,369 | 258 | 199 | 77.13 | 7/6/0 | 1 | 27/1/8 | 6 |
| 32 | 15,888 | 563 | 497 | 88.28 | 18/10/1 | 14 | 29/0/37 | 17 |
| 33 | 16,767 | 248 | 186 | 75.00 | 7/4/1 | 0 | 28/0/34 | 7 |
| 34 | 17,229 | 227 | 167 | 73.57 | 6/4/0 | 2 | 27/0/33 | 6 |
| 35 | 17,522 | 114 | 0 | 0.00 | 0/0/0 | 0 | 114/0/0 | - |
| 36 | 17,695 | 143 | 92 | 64.34 | 3/2/0 | 0 | 27/0/24 | 2 |
| 37 | 18,019 | 283 | 217 | 76.68 | 8/5/0 | 0 | 28/1/31 | 8 |
| 38 | 18,670 | 392 | 330 | 84.18 | 10/6/1 | 1 | 30/0/32 | 9 |
| 39 | 19,279 | 328 | 267 | 81.40 | 9/6/1 | 1 | 28/0/33 | 9 |
| 40 | 19,885 | 237 | 167 | 70.46 | 8/3/1 | 3 | 20/2/39 | 7 |
| 41 | 20,423 | 336 | 275 | 81.85 | 10/8/1 | 4 | 27/0/34 | 8 |

## 2. Prior hypotheses: confirm/refute at scale

### Ledger

| Prior claim | 11-minute evidence | Ruling |
|---|---|---|
| All selected bounces are exactly z=radius | 161/170 at radius; 9 above it, all flagged `out_of_sequence`. All 154/154 in-sequence bounces are at radius. | **REFUTED literally; CONFIRMED for in-sequence anchors** |
| Shot and bounce anchors reproject at ~0 | Bounce: 154/154 in-sequence <=1e-6 px. Shot: 245/260 in-sequence <=1e-6 px; 15 nonzero first-shot/serve anchors. | **REFUTED as universal; typed subpolicies exist** |
| Events partition piecewise flight | Event delta-v median 12.152 m/s vs interior 0.311 m/s = 39.06x; clean-triplet ratio 41.24x. Per-rally ratio exists in all 41 emitting rallies. | **CONFIRMED method family** |
| Selection is not action argmax | 6,118/7,808 = 78.356%, close to prior 80.87%; ball-only is 76.96%, while shot/bounce are 100%. | **CONFIRMED** |
| Explicit interpolation is rare | 95/7,808 = 1.217%, still rare but 2.23x the prior 0.546%. | **CONFIRMED, reweighted upward slightly** |
| Every evaluable piecewise segment is ballistic under the diagnostic | 392/422 = 92.89%; clean-event segments 364/388 = 93.81%; 30 fail. | **REFUTED literally; strong majority signature remains** |
| Longer arcs do not identify drag | 130 eligible; 2 local signatures; both fail physics plausibility; 0 pass-compatible signatures. Median drag `7.08e-6 /m`, median RMSE gain `0.00012%`, median delta-BIC `-4.36`. | **CONFIRMED** |
| Radius is a useful depth cue | 7,178 pairs; pooled Pearson/Spearman/R2 `0.535/0.797/0.287`; all 41 rallies positive, all 41 Spearman >=0.5. | **CONFIRMED as cue; REFUTED as one universal strong linear law** |

### Anchor reprojection detail

| Selected kind / slice | n | <=1e-6 px | p50 px | p95 px | max px |
|---|---:|---:|---:|---:|---:|
| Bounce, in-sequence | 154 | 154 | ~0 | ~0 | `1.72e-12` |
| Bounce, out-of-sequence | 16 | 7 | 8.21 | 72.12 | 79.87 |
| Shot, in-sequence | 260 | 245 | ~0 | 14.08 | 217.17 |
| Shot, out-of-sequence | 32 | 5 | 34.52 | 376.15 | 7,731.63 |
| Shot, first shot of rally | 41 | - | ~0 | 84.20 | 217.17 |
| Ordinary ball | 7,318 | 0 | 15.61 | 85.10 | 19,150.50 |

All 15 nonzero in-sequence shot anchors are first-shot/serve anchors. That is
evidence of a separate initialization policy (many have z around 3 ft), not
proof of its exact implementation. The very large out-of-sequence and ordinary
tail also reinforces that `selected` does not mean ray-fit or accurate.

### Velocity-discontinuity distribution

| Slice | n | p50 m/s | p95 m/s | max m/s |
|---|---:|---:|---:|---:|
| Event boundaries | 396 | 12.152 | 24.539 | 1,234.442 |
| Interior | 7,296 | 0.311 | 0.446 | 271.550 |
| Per-rally event/interior ratio | 41 | 45.157 | 63.334 | 70.991 |

The global ratio is 39.06x; excluding any three-frame stencil containing an
exported out-of-sequence event gives 41.24x. Extreme maxima are retained as raw
observations and are not used to make the method-family ruling.

### Physics re-integration

The frozen rule remains RMSE <=0.15 m, p95 <=0.30 m, observed speed <=35 m/s,
and z >=-0.05 m. Of 490 event-partitioned segments, 422 are evaluable, 63 have
fewer than four samples, and 5 have insufficient duration.

| Rally | Segments | Evaluable | Pass | Pass % | Clean eval/pass |
|---:|---:|---:|---:|---:|---:|
| 0 | 4 | 3 | 3 | 100.0 | 3/3 |
| 1 | 3 | 3 | 3 | 100.0 | 3/3 |
| 2 | 6 | 5 | 4 | 80.0 | 4/5 |
| 3 | 8 | 6 | 5 | 83.3 | 5/6 |
| 4 | 15 | 11 | 11 | 100.0 | 11/11 |
| 5 | 7 | 6 | 6 | 100.0 | 6/6 |
| 6 | 6 | 5 | 4 | 80.0 | 4/5 |
| 7 | 19 | 17 | 16 | 94.1 | 15/16 |
| 8 | 16 | 15 | 14 | 93.3 | 13/14 |
| 9 | 10 | 8 | 5 | 62.5 | 5/7 |
| 10 | 11 | 10 | 9 | 90.0 | 9/10 |
| 11 | 6 | 5 | 5 | 100.0 | 5/5 |
| 12 | 9 | 7 | 7 | 100.0 | 7/7 |
| 13 | 6 | 5 | 5 | 100.0 | 5/5 |
| 14 | 16 | 15 | 15 | 100.0 | 15/15 |
| 15 | 16 | 15 | 15 | 100.0 | 15/15 |
| 16 | 7 | 5 | 4 | 80.0 | 2/3 |
| 17 | 10 | 9 | 9 | 100.0 | 9/9 |
| 18 | 4 | 3 | 3 | 100.0 | 3/3 |
| 19 | 7 | 5 | 4 | 80.0 | 4/5 |
| 20 | 9 | 7 | 6 | 85.7 | 6/7 |
| 21 | 29 | 26 | 24 | 92.3 | 21/22 |
| 22 | 10 | 8 | 6 | 75.0 | 5/7 |
| 23 | 6 | 5 | 5 | 100.0 | 5/5 |
| 24 | 15 | 13 | 12 | 92.3 | 11/12 |
| 25 | 6 | 5 | 5 | 100.0 | 5/5 |
| 26 | 33 | 29 | 29 | 100.0 | 23/23 |
| 27 | 10 | 9 | 6 | 66.7 | 6/9 |
| 28 | 4 | 3 | 3 | 100.0 | 3/3 |
| 29 | 28 | 25 | 23 | 92.0 | 19/19 |
| 30 | 8 | 7 | 5 | 71.4 | 4/5 |
| 31 | 13 | 10 | 10 | 100.0 | 9/9 |
| 32 | 29 | 28 | 27 | 96.4 | 25/25 |
| 33 | 12 | 10 | 10 | 100.0 | 10/10 |
| 34 | 10 | 9 | 9 | 100.0 | 9/9 |
| 35 | 0 | 0 | 0 | - | 0/0 |
| 36 | 5 | 4 | 4 | 100.0 | 4/4 |
| 37 | 13 | 11 | 11 | 100.0 | 11/11 |
| 38 | 17 | 16 | 15 | 93.8 | 14/15 |
| 39 | 16 | 14 | 14 | 100.0 | 14/14 |
| 40 | 12 | 9 | 7 | 77.8 | 6/8 |
| 41 | 19 | 16 | 14 | 87.5 | 11/13 |
| **Total** | **490** | **422** | **392** | **92.89** | **364/388 (93.81%)** |

All 490 per-segment rows, including the 30 failures and 68 non-evaluable
segments, are preserved in
`forensics_metrics.json#/hypotheses/physics_reintegration/segments`.

## 3. Camera over 11 minutes

| Metric | Result |
|---|---:|
| `cameraSegments` | 1 |
| Segment frame span | 0..20,921 inclusive (the full 20,922-frame export) |
| FOV | 1.509993462 rad (86.516 deg) |
| Position ft | `[27.27658, 51.01127, 5.05095]` |
| Orientation rad | pitch -0.185428; roll -0.0000138; yaw -2.428380 |
| Court points | 12 |
| Court-point confidence min / p50 / p95 / max | 0.946423 / 0.963478 / 0.976621 / 0.978721 |
| Court-point spread min / p50 / p95 / max | 0.000194 / 0.001537 / 0.002043 / 0.002082 |
| Between-segment drift rows | 0; not measurable because there is only one solve |

There are no re-solves, no per-frame camera poses, and no exported optical-flow
or stabilization state. Therefore the export supports **one static solve** and
provides **no positive evidence of wobble/motion tracking**. It cannot prove
that the physical camera never moved or that no unexported stabilization ran.
Court-point confidence/spread describe one aggregate solve, so temporal drift
cannot be inferred from their range across the 12 spatial points.

## 4. Radius depth cue at scale

### Full export

| Slice | n | Pearson r | Spearman rho | Linear R2 | Relative residual p50 / p95 |
|---|---:|---:|---:|---:|---:|
| Radius vs inverse camera depth | 7,178 | 0.535 | 0.797 | 0.287 | 5.45% / 38.70% |
| Radius vs inverse Euclidean camera distance | 7,178 | 0.523 | 0.765 | 0.274 | 6.85% / 39.92% |
| Within-rally standardized radius vs inverse depth | 7,178 | 0.635 | 0.769 | 0.404 | diagnostic only |
| Prior one-rally export | 168 | 0.850 | 0.774 | 0.722 | 7.04% p50 |

Pooling all rallies weakens linear fit substantially but leaves rank ordering
strong. Within-rally standardization recovers some linear relationship, which
argues for rally/arc-specific nuisance calibration instead of one global
radius-to-depth line.

### Per-rally variance

All 41 emitting rallies meet the >=10-pair criterion. Pearson is positive in
41/41, Pearson >=0.5 in 33/41, Spearman >=0.5 in 41/41, and R2 >=0.25 in
33/41. Pearson across rallies is min/p50/p95/max
`0.104/0.776/0.953/0.985`; Spearman is `0.567/0.789/0.949/0.968`; R2 is
`0.011/0.602/0.909/0.971`.

| Rally | n | Pearson r | Spearman rho | R2 |
|---:|---:|---:|---:|---:|
| 0 | 52 | 0.953 | 0.867 | 0.909 |
| 1 | 43 | 0.874 | 0.768 | 0.764 |
| 2 | 104 | 0.979 | 0.913 | 0.958 |
| 3 | 112 | 0.175 | 0.831 | 0.031 |
| 4 | 212 | 0.776 | 0.761 | 0.602 |
| 5 | 120 | 0.935 | 0.928 | 0.873 |
| 6 | 101 | 0.620 | 0.735 | 0.385 |
| 7 | 288 | 0.849 | 0.811 | 0.721 |
| 8 | 243 | 0.747 | 0.764 | 0.559 |
| 9 | 121 | 0.856 | 0.789 | 0.732 |
| 10 | 217 | 0.762 | 0.847 | 0.580 |
| 11 | 100 | 0.925 | 0.949 | 0.856 |
| 12 | 160 | 0.271 | 0.873 | 0.074 |
| 13 | 92 | 0.736 | 0.792 | 0.541 |
| 14 | 270 | 0.636 | 0.715 | 0.405 |
| 15 | 251 | 0.896 | 0.710 | 0.803 |
| 16 | 51 | 0.614 | 0.760 | 0.377 |
| 17 | 138 | 0.700 | 0.739 | 0.490 |
| 18 | 56 | 0.750 | 0.762 | 0.562 |
| 19 | 105 | 0.856 | 0.927 | 0.733 |
| 20 | 131 | 0.935 | 0.848 | 0.874 |
| 21 | 379 | 0.215 | 0.709 | 0.046 |
| 22 | 137 | 0.934 | 0.890 | 0.873 |
| 23 | 98 | 0.791 | 0.968 | 0.626 |
| 24 | 215 | 0.897 | 0.878 | 0.804 |
| 25 | 81 | 0.985 | 0.968 | 0.971 |
| 26 | 411 | 0.374 | 0.664 | 0.140 |
| 27 | 190 | 0.789 | 0.779 | 0.622 |
| 28 | 51 | 0.947 | 0.914 | 0.898 |
| 29 | 328 | 0.104 | 0.567 | 0.011 |
| 30 | 114 | 0.375 | 0.694 | 0.141 |
| 31 | 191 | 0.803 | 0.858 | 0.645 |
| 32 | 441 | 0.450 | 0.632 | 0.203 |
| 33 | 184 | 0.615 | 0.781 | 0.378 |
| 34 | 156 | 0.921 | 0.798 | 0.849 |
| 36 | 81 | 0.947 | 0.933 | 0.897 |
| 37 | 213 | 0.846 | 0.878 | 0.716 |
| 38 | 304 | 0.558 | 0.653 | 0.311 |
| 39 | 246 | 0.660 | 0.746 | 0.435 |
| 40 | 152 | 0.743 | 0.742 | 0.552 |
| 41 | 239 | 0.156 | 0.611 | 0.024 |

**Ruling:** the cue holds universally in sign and monotonically on this export,
not universally as a strong linear metric. Radius should be an independent,
per-arc calibrated residual with robust weighting and abstention. Correlation
does not prove pb.vision used it causally.

## 5. Product surface map: insights + stats versus our facts builder

PB ships 41 product rallies, 256 shot records, 23 highlights, four player
advice lists, an 11-key per-player insights object, a 120-key dynamic aggregate
matrix, and four stats player objects whose union has 50 top-level fields.

Our authoritative audited fact types are currently `rally`, `movement`,
`positioning`, and `recovery`; they are `user_facing=false`, authority
`preview`, and gate status `unpassed`. The legacy metrics object also computes
avg/p95 speed, zone occupancy, kitchen proximity, and cue-level contact counts,
but cue-level contact is not an audited user-authority fact.

| Fact class | pb.vision field paths | Our status | Exact boundary |
|---|---|---|---|
| Session identity and game summary | `session.*`; `game_data.*`; `stats.json:game.*` | **lacks** | Source run identity exists, but no score, game result, or summary fact. |
| Rally timing/duration | `rallies[].{start_ms,end_ms}` | **covers preview** | Audited `rally_duration_s` with interval and lineage. |
| Rally result/scoring | `rallies[].winning_team`; `.scoring_info.{running_score,server_number,likely_bad}` | **lacks** | No winner/server/score fact. |
| Shot identity/timing/hitter | `rallies[].shots[].{player_id,start_ms,end_ms,is_final}` | **gate-blocked** | Shot omission; macro-F1 >=0.65 and top-2 >=0.85 required. |
| Shot taxonomy/paddle side | `.shots[].{shot_type,stroke_type,stroke_side,vertical_type,is_volley,is_reset,is_speedup,is_poach,is_putaway,is_passing}`; `.tags.*` | **gate-blocked** | No audited shot fact; same shot gate. |
| Shot kinematics | `.resulting_ball_movement.{speed,distance,height_over_net,crossed_net,is_volleyed}`; `.angles.{yaw,pitch,direction}` | **gate-blocked** | NS 3.3 requires BALL+CAL+event for apex/net/landing authority. |
| 3D trajectory/apex/landing | `.trajectory.{confidence,start,peak,end}`; endpoints `{ms,location.{x,y,z},zone}` | **gate-blocked** | Landing omission requires `ball_arc_solved`, CAL, refined events, and BALL/CAL/event gates. |
| Result/fault/in-out | `.shots[].{winner_type,errors.unforced}`; `.errors.faults.{net,short,kitchen,excess_bounce,paddle_hit_net,out}`; `.out.{outcome,direction,side}` | **gate-blocked** | No audited in/out fact; same upstream gates. |
| Shot quality/tactics | `.quality.{overall,execution,selection,pressure}`; `{advantage_scale,shooter_positioning_score,partner_positioning_score,shooter_movement_from_last_shot}` | **lacks** | Position facts do not implement quality, pressure, advantage, or causality. |
| Player movement/coverage | `player_data[].court_coverage.*`; `stats.players[].total_distance_covered`; `insights.stats.*ft_moved*` | **partial preview** | Covers gap-aware distance and avg/p95 speed; lacks heat map and session aggregates. |
| Positioning/spacing/kitchen | `rallies[].players[].{started_on_left_side,kitchen_arrivals,had_arrival_opportunity}`; `.team_stats.*`; `player_data[].{left_side_percentage,kitchen_arrival_percentage,team_kitchen_arrival,positional_performance,role_data}` | **partial preview** | Covers dominant zone, occupancy, kitchen proximity, first return-to-kitchen; lacks team/role/opportunity scores. |
| Contact | `cv_export:balls.shot.player`; `rallies[].shots[].player_id` | **gate-blocked** | Legacy count may be an unverified cue; audited contact omitted pending BODY+contact gates. |
| Highlights | `highlights[].{rally_idx,shot_start_idx,shot_end_idx,s,e,kind,score,short_description,rally_ending}`; `.events[]` | **lacks** | No highlight selection or clip fact. |
| Coach advice | `coach_advice[].advice[].{kind,relevance,value,ci,method}` | **lacks** | NS-05.2 comparator and NS-05.3 language layer unbuilt. |
| Trends/ratings | `player_data[].trends.{ratings,shot_accuracy,shot_quality,shot_selection,serve_depth,return_depth,serve_speed,kitchen_arrivals}.*` | **lacks** | No session trend/rating builder or self-relative comparator. |
| Per-player shot aggregates | `stats.players[].<bucket>.{count,average_quality,average_height_above_net,average_baseline_distance,speed_stats,outcome_stats}` | **gate-blocked** | Shot/landing/result inputs unpassed; no per-player shot aggregate builder. |
| Global feature matrix | `insights.stats.<dynamic_metric>[player]` | **lacks** | No equivalent 120-key session matrix. |

The stats player bucket union includes serve/return/third/fourth/fifth,
third-drive/drop/lob, dink/drive/drop/reset/speedup/smash, forehand/backhand,
poach/passing, court-area/side splits, ball directions, role/team kitchen
arrival, distance, quality, fault rates, final shots, and team rally-length
outcomes. Exact 50-key union and all nested paths are in the generated JSON
and schema inventory.

## 6. In/out and landing calls

| Surface | Count / representation |
|---|---|
| Shot trajectory endpoints | 256/256 with `trajectory.end.{ms,location{x,y,z},zone}` |
| End zones | deep 26; kitchen 82; mid 60; net 24; out 4; short 60 |
| Structured out faults | 5 at `errors.faults.out` |
| Landed out | 3: two `{outcome:landed,direction:long}`, one `{landed,left,far}` |
| Intercepted predicted-out | 2: `{outcome:intercepted,direction:long}` while endpoint zone is not out |
| Corrected endpoint locations | 3 with `trajectory.end.location.corrected=true` |
| Out-zone endpoint with no `faults.out` | 1 |
| General trajectory confidence | 256; min/p50/max 0.280/0.869/0.976 |
| Line-call-specific confidence fields | none |
| Too-close/abstention values | none |

PB represents a landing/end estimate with 3D location and a coarse zone. An
out fault optionally adds `outcome`, `direction`, and sometimes `side`.
`outcome=intercepted` is a projected-out semantic rather than an observed
landing. The general trajectory confidence cannot be relabeled as line-call
confidence. There is no exported `too_close_to_call`, uncertainty-to-line,
margin, covariance, or explicit abstention analog. The 4 end-zone-out vs 5
structured-out mismatch confirms that consumers must not infer a single
canonical line-call field from this export.

## 7. Implications and updated reproduction map

The 11-minute evidence increases confidence in the event-anchored/ballistic
method family, but it also reveals explicit exceptional policies
(`out_of_sequence`, first-shot initialization, missing windows). That makes
more state-search complexity a lower priority than evidence quality and robust
policy handling.

| New rank | Prior rank | Reproduction item | Ruling / reason |
|---:|---:|---|---|
| 1 | 4 | Audio ordering + BlurBall/WASB blur event-anchor evidence | **Up.** Scale retains sharp event discontinuities; TT3D/recovery cannot manufacture accepted boundaries. |
| 2 | 3 | Ball-radius depth residual with per-rally calibration/abstention | **Up.** Directional cue holds in all 41 rallies; global linear strength is variable, so robust ablation is required. |
| 3 | 5 | Reviewed-anchor outlier quarantine + a new two-ended robust fit | **Conditional.** Prior combined pinning regressed; only a distinct bounded fit after reviewed anchors avoids repeating it. |
| 4 | 1 | Adjacent-accepted-fit UKF/physics recovery, predicted provenance | **Down/blocked.** `ball_recovery` produced 58/252 on every arm: zero PB-rally coverage gain. Retry only after accepted boundaries improve. |
| 5 | 2 | TT3D joint anchor-state search | **Rejected.** TT3D 9/13 and TT3D+pinning 8/13 missed `<5/11`; reprojection/tails regressed. |
| 6 | 6 | Whole-rally DP segmentation | **Killed.** The preregistered TT3D follow-on was explicitly killed; no launch without a genuinely new hypothesis. |

### Exact next three engineering moves

| Order | Move | Measurable target |
|---:|---|---|
| 1 | Use the now-available source video to freeze identity/alignment and run the current stack once before tuning. | Preserve source SHA-256 `272a...8383` and encoded PTS; reconcile the verified 20,922-frame/30fps grid or explain every mismatch; map 42 CV/41 insight rallies; save one immutable ours-vs-PB scorecard plus a human-reviewed union of PB and our event candidates. PB remains reference-only. |
| 2 | Fix event evidence before another state search: correct audio order and add blur-aware shot/bounce proposals. | On the frozen reviewed same-clip set: precision >=0.90 and recall >=0.90 within +/-2 frames; contact timing p90 <=40 ms; zero frozen 2D BALL F1/hFP/tail regression; >=80% of reviewed flight intervals have both accepted endpoint classes. |
| 3 | Ablate radius depth plus a new robust two-ended fit; re-enable recovery only behind accepted-fit gates. | Supported size residuals on >=70% of frames where our diameter is measurable; >=20% lower leave-one-anchor-out depth p95 with no worse held-out reprojection/bounce/endpoint tail; recovered samples `physics_predicted`; 100% declared physics pass; zero >35 m/s steps; coverage gain without measured-label substitution. |

The same-clip targets are engineering gates. No comparator-agreement number is
an accuracy promotion. Actual 3D landing/in-out promotion still requires
independent surveyed/multiview truth under NS-02/03.

## Honest issues

- pb.vision is not ground truth. Agreement, mismatch, coverage, and its product
  decisions are diagnostic only.
- The original one-rally comparator did not apply unchanged: it crashed on this
  export because a velocity seed exceeded optimizer bounds. The lane copy
  fixes scale operation by isolating rallies and clipping only the initializer;
  raw observations remain untouched.
- The image size is now verified from the 1280x720 source; the Euler convention
  remains inferred. Exact large subsets strongly support it, but ordinary and
  exceptional tails may include unexported distortion or a separate internal
  projection policy.
- `selected` and `interpolated=false` are not measurement proof. The export
  contains low-confidence, out-of-sequence, non-ray-fit, and physically failing
  selected rows.
- The bounce-radius conclusion must preserve flags: 9/170 exceptions refute
  “all,” while 154/154 confirms only the in-sequence subset.
- Physics re-integration is self-consistency, not accuracy. A wrong trajectory
  can integrate cleanly; 30 segments fail even this diagnostic.
- Drag identifiability uses a local i.i.d.-residual approximation. Its two
  nominal signatures occur on physics-failing segments and are not physical
  parameter evidence.
- Radius correlation does not prove causal use by pb.vision, and its per-rally
  linear strength varies from R2 0.011 to 0.971.
- One camera segment means one exported static solve; it cannot rule out
  physical wobble or unexported stabilization.
- Insights has no line-call confidence/abstention field and contains one
  out-zone endpoint without `faults.out`; do not invent certainty or canonical
  semantics.
- Our product-surface coverage is code/schema coverage only. Current audited
  facts remain `user_facing=false`, preview/unpassed; `VERIFIED=0` is binding.
- The source video arrived after the export analysis began. It is now hashed and
  frame-verified, but no same-clip DinkVision pipeline run was added to this
  export-forensics lane.
- No source, data, config, best-stack, branch, commit, or staged-file change was
  made. Best-stack delta: **(c) none**.
