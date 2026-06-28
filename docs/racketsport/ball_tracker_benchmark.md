# Ball Tracker Benchmark

The sparse ball clicks in `runs/eval0/prototype_gate_h100_v2/ball_click_review_30/`
are held-out prototype review labels. They must not be read by candidate
trackers, post-processors, or production runtime paths.

## Candidate Rules

- Generalizable candidates may read the source video, calibration artifacts,
  model outputs, and non-click metadata.
- Generalizable candidates must not read `ball_points.json`, click summaries, or
  click-corrected ball tracks.
- Click-corrected tracks may be scored only as `oracle_not_generalizable`; they
  are useful for debugging the upper bound, not for picking a production path.
- Benchmark outputs are `scored_not_gate_verified` until a real acceptance gate
  is defined and passed on a representative labeled set.

## Current Prototype Candidates

- `tracknet_raw`: raw TrackNetV3 output.
- `tracknet_court_120px`: raw TrackNet filtered to the calibrated target court.
- `tracknet_court_temporal_outlier`: target-court output with isolated
  impossible jumps removed.
- `tracknet_court_temporal_path`: target-court output reduced to the longest
  motion-consistent path, with only short gaps filled.
- `oracle_click_corrected`: sparse-click identity filter output. Excluded from
  production ranking because it consumes held-out labels.

## Current Read

The benchmark at
`runs/eval0/prototype_gate_h100_v2/ball_tracker_benchmark/benchmark_summary.md`
shows that court filtering and temporal filtering reduce background balls and
large random jumps, but they do not solve the base detector issue. The strict
temporal path improves stability and hidden-frame behavior while losing too much
visible recall. The less destructive outlier filter keeps recall but leaves too
many hidden-frame false positives.

The next model-side candidate should be a sports-ball-specific tracker with
motion attention, starting with TrackNetV4 if usable weights or training data are
available. Generic point trackers should be evaluated only with automatic seeds
from a detector or motion model; human-click seeded point tracking is another
oracle variant, not a production-comparable tracker.
