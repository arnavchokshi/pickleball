# Observation logs — ball_lane_20260723 (A-5)

Measurement-only dataset artifacts; `VERIFIED=0` binding. Built with
`scripts/racketsport/build_ball_observation_log.py` in the A-3 contract format
(`threed/racketsport/ball_metric3d_contract.py`, `schema_version` 1,
`artifact_type` `racketsport_ball_solver_observation_log`).

## burlington.observation_log.json

Clip: `burlington_gold_0300_low_steep_corner` (logged under the short name
`burlington`; the artifact's own identity is preserved in the log's
`source_clip_id` field, read from the solved artifact's `clip_id`). 600
frames: 479 observed (pixel + world ray), 121 missing; verdicts 181 accepted
/ 407 rejected_fail_closed / 12 hidden — consistent with
`runs/ball_lane_20260723/characterization/report.json` for the same solve
(confident 3D coverage 181, hidden 12). Note: `accepted` means the owning
segment is trusted and the frame carries a solver world position — it
includes solver-interpolated frames with no detector observation.

Output sha256 (byte-identical across re-runs):
`9016c7795cfbda453dd4160320313552a51c64a8078de8dc5065b139412f59f2`

(Regenerated 2026-07-24 to add `source_clip_id`; verified against the prior
artifact `ccb57dee…f24cd` that the ONLY byte change is the added
`"source_clip_id": "burlington_gold_0300_low_steep_corner"` line — all 600
frames, inputs, and every other byte are identical, and the new bytes are
again stable across re-runs.)

Source artifacts (main-checkout paths, sha256):

| kind | path | sha256 |
|---|---|---|
| ball_track_arc_solved | `runs/lanes/ball_p3a_bvp_solver_20260705/three_clip_default_chain/burlington_gold_0300_low_steep_corner/ball_track_arc_solved.json` | `df7a6a835946b6891a0d536d6b438c106fd1d3232c6d29f1712ddef30266cad6` |
| ball_chain_manifest | `runs/lanes/ball_p3a_bvp_solver_20260705/three_clip_default_chain/burlington_gold_0300_low_steep_corner/ball_chain_manifest.json` | `f5638b1a536c1362477a8049886e6a0d06ea1c541b33314c96bd4c15b895fe03` |
| court_calibration | `runs/lanes/ball_f1_three_clip_runs_20260705/burlington_gold_0300_low_steep_corner/court_calibration.json` | `aec27ecc0f377ee930f363173032316332811ac573199a633ebe499747abed1d` |

Calibration provenance: the characterization manifest listed a holdout-lane
calibration copy whose sha did NOT match the solve's recorded input
(`calibration_sha_matches_solver_input: false`), so it could not be used for
ray computation (fail closed). This log instead uses the ORIGINAL
`ball_f1_three_clip_runs_20260705` calibration, which byte-matches the
`court_calibration` sha recorded in `ball_chain_manifest.json`
(`calibration_sha_verified: true` in the log), so world rays are computed
from the exact calibration bytes the solve consumed.

Rebuild command (from repo root; paths relative to the main checkout):

```bash
.venv/bin/python scripts/racketsport/build_ball_observation_log.py \
  --clip burlington=runs/lanes/ball_p3a_bvp_solver_20260705/three_clip_default_chain/burlington_gold_0300_low_steep_corner \
  --calibration burlington=runs/lanes/ball_f1_three_clip_runs_20260705/burlington_gold_0300_low_steep_corner/court_calibration.json \
  --out-dir runs/ball_lane_20260723/observation_logs
```

## FOLLOWUPS (reviewer minors, documented not yet implemented)

1. **Unknown-key policy (contract)**: `from_json_dict` validates known fields
   but silently ignores unrecognized extra keys in payloads. Decide and
   enforce a policy (reject vs warn vs allow-listed extensions) before any
   second producer writes these artifacts.
2. **CLI loudness**: `build_ball_observation_log.py` collapses all failures
   to a single stderr line + exit 2. Fine for tests; for lane use it should
   distinguish per-clip failures (continue vs abort) and echo which clip
   failed in multi-clip runs.
3. **`_relative_posix` fallback**: when a source artifact lies outside
   `--root`, the recorded provenance path silently falls back to an absolute
   path, which breaks the "no absolute paths in artifact bytes" determinism
   stance across machines. Should either fail loudly or require an explicit
   opt-in. (The committed burlington log is unaffected: all inputs resolve
   under the main-checkout root.)
