# Observation logs — ball_lane_20260723 (A-5)

Measurement-only dataset artifacts; `VERIFIED=0` binding. Built with
`scripts/racketsport/build_ball_observation_log.py` in the A-3 contract format
(`threed/racketsport/ball_metric3d_contract.py`, `schema_version` 1,
`artifact_type` `racketsport_ball_solver_observation_log`).

## burlington.observation_log.json

Clip: `burlington_gold_0300_low_steep_corner` (logged under the short name
`burlington`). 600 frames: 479 observed (pixel + world ray), 121 missing;
verdicts 181 accepted / 407 rejected_fail_closed / 12 hidden — consistent
with `runs/ball_lane_20260723/characterization/report.json` for the same
solve (confident 3D coverage 181, hidden 12).

Output sha256 (byte-identical across re-runs):
`ccb57dee2ba0dc3aa7ecae271f6c9b5f70ad7a10520bfeab2cd684c8a01f24cd`

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
