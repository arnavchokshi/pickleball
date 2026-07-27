# TT3D external validation harness

The only falsifiable 3D ball accuracy measurement in this repo that uses data we
did not produce ourselves. Written by lane `tt3d_external_validation_20260726`;
documented here because it had no entry anywhere outside a `runs/` lane report,
which made `scripts/racketsport/audit_dead_code.py` report two of its three
files as unreferenced surfaces.

**They are not dead code.** See "Why this file exists" at the bottom.

## Licence constraint (binding)

The upstream TT3D repository (Gossard/Ziegler/Zell, Univ. Tübingen, CVPR 2025
CVSports — <https://github.com/cogsys-tuebingen/tt3d>) carries **no LICENSE
file**. This data is **internal validation only**: never train on it, never ship
it, and never admit it to the product data ledger as training data. Both source
files repeat this at the top; keep it that way.

## The three files

| file | role |
| --- | --- |
| `scripts/tt3d/tt3d_adapter.py` | Coordinate/unit adapter mapping the TT3D evaluation set onto DinkVision solver conventions — Rodrigues rotation, world→camera, the TT3D Z-up table frame, ball mass/diameter/drag constants. The dangerous part of the exercise: a silent unit, frame, or sign error here yields a confidently wrong accuracy number. |
| `scripts/tt3d/run_tt3d_validation.py` | CLI driver for experiments E0–E3 (see below). Writes `runs/lanes/tt3d_external_validation_20260726/report.json`. |
| `scripts/tt3d/sigma_calibration.py` | Scores the bounce-anchor uncertainty model against measured error, LEGACY isotropic scalar vs RAY-aligned anisotropic, in one run so before/after is apples to apples. Writes `sigma_calibration.json`. |

### Experiments in `run_tt3d_validation.py`

- **E0** coordinate-mapping gate: project TT3D ground-truth 3D through our camera
  model and compare with the dataset's own `(u, v)`. Must be ~0 px or everything
  downstream is meaningless.
- **E1** bounce-anchor accuracy: `build_bounce_anchor` (pixel ray × table plane)
  against ground-truth contact position, plus `anchor_sigma_for_bounce`.
- **E2** monocular 3D trajectory: `fit_weak_flight_segment` seeded by the bounce
  anchor, scored in 3D with a camera-depth vs image-plane decomposition.
- **E3** analytic depth-blindness demonstration: slide a point along its own
  camera ray — reprojection error stays 0 while 3D error grows without bound.

## Running it

```
PYTHONPATH=. python3 scripts/tt3d/run_tt3d_validation.py
PYTHONPATH=. python3 scripts/tt3d/sigma_calibration.py
```

Both read `data/external/tt3d_repo/data/evaluation`, a clone of the upstream
repo (~4.6 MB of evaluation data). **That directory is gitignored and is not
present in every checkout.** As of 2026-07-27 the only local copy sits in the
`tt3d-validate-20260726` worktree; re-clone from the URL above if it is absent.

## Test coverage

`tests/tt3d/test_tt3d_adapter.py` pins the adapter — Rodrigues identity,
round-trip world↔camera, and the projection agreement that E0 gates on. It
`skipif`s itself when the evaluation data is absent, so a green run in a
checkout without `data/external/` proves nothing about the adapter.

Note that this test lives under `tests/tt3d/`, not `tests/racketsport/`.

## What it measured

`runs/lanes/tt3d_external_validation_20260726/` holds `report.json`,
`sigma_calibration.json` and `data_manifest.sha256`. That evidence is cited by
the `ball.bounce_anchor_uncertainty` entry in
`configs/racketsport/best_stack.json`, which records `sport: "table tennis, NOT
pickleball"`, `pickleball_gate_passed: false` and
`verified_status_for_pickleball: 0`. **VERIFIED=0 for pickleball**: a table
tennis result does not transfer, and the entry says so.

## Why this file exists

`scripts/racketsport/audit_dead_code.py` classifies a Python source as
`referenced` when some tracked text file names its path, names its module, or
imports it, or when a matching test file exists under `tests/racketsport/`.
`scripts/tt3d/run_tt3d_validation.py` and `scripts/tt3d/tt3d_adapter.py` matched
none of those:

- the adapter's only test is at `tests/tt3d/test_tt3d_adapter.py`, and the
  auditor's test-matching scans `tests/racketsport/` only;
- the driver is a CLI nobody imports, and its only prose write-up lives under
  `runs/`, which the auditor ignores by design.

Deleting them would also orphan `scripts/tt3d/sigma_calibration.py`, whose sole
inbound reference is a comment inside `scripts/tt3d/run_tt3d_validation.py` —
the three files stand or fall together. Documenting the harness is the correct
resolution; removing it would destroy the only external-data accuracy
measurement the ball work has.
