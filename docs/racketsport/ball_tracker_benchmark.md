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
- `tracknet_court_local_search`: target-court output post-processed with
  bounded CPU pixel evidence around the predicted trajectory.
- `fusion_temporal_vball`: target-court TrackNet plus temporal backbone fused
  with independent VballNet verifier tracks. This does not use click labels.
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

The CPU local-search candidate is not currently a winner: it reduces large
jumps and can recover some visible labels, but it also increases hidden-frame
false positives on the accepted four-clip benchmark. Keep it as an experimental
diagnostic path until its evidence model can distinguish real target-court balls
from court lines, paddle flashes, and background balls.

The best current no-click prototype is the TrackNet/VballNet fusion pass. It
uses TrackNet for recall, the temporal path as a stable backbone, and VballNet
as an independent verifier. On the accepted four-clip held-out benchmark it
beats the previous temporal-path score while keeping p90 error and teleports
far below raw TrackNet. It is still `fused_not_gate_verified`, not a BALL gate.

The next model-side candidate should be a sports-ball-specific tracker with
motion attention, starting with TrackNetV4 if usable weights or training data are
available. The H100 has the official TrackNetV4 repo cloned, but the upstream
`docs/RESULT.md` checkpoint links are placeholders in the public repo snapshot,
so a real run still needs weights or fine-tuning. Generic point trackers should
be evaluated only with automatic seeds from a detector or motion model;
human-click seeded point tracking is another oracle variant, not a
production-comparable tracker.

A usable adjacent pretrained model was found in `asigatchov/vball-net`: the
Google Drive demo archive contains VballNet Keras and ONNX weights. Those
weights are volleyball-trained, so they are not a substitute for official
TrackNetV4 or pickleball fine-tuning, but they provide a useful independent
motion-attention verifier for fusion experiments.
