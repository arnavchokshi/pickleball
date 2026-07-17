# `placement_trajectory_refined.json` schema

Schema version 1. This is a preview-band, `VERIFIED=0` refinement artifact. It is
also skeleton3d-shaped so the frozen BODY scorer can consume it without an
adapter. Coordinates are metres in legacy `court_Z0`, typed as
`world_court_netcenter_z_up_m`: net-centre origin, court-width +X, court-length
+Y, and +Z up. The refiner performs no image transform, so distortion state is
not applicable.

## Top level

- `schema_version`: integer `1`.
- `artifact_type`: `placement_trajectory_refined`.
- `world_frame`: `court_Z0`.
- `coordinate_space`: `world_court_netcenter_z_up_m`.
- `preview_band`: always `true`.
- `VERIFIED`: always `0` for this internal-card evidence.
- `fps`, `joint_names`, `source_model`, and other inherited skeleton metadata:
  copied unchanged from the immutable `skeleton3d.json` input.
- `players`: refined skeleton-shaped player rows described below.
- `placement_trajectory_refinement`: global solve metadata, provenance, policy,
  configuration, and summaries.

## `players[].frames[]`

Inherited `frame_idx`, `t`, and `joint_conf` are unchanged. `transl_world` and
every point in `joints_world` receive exactly the same rigid XYZ correction;
therefore root-relative pose is unchanged.

Each frame adds `placement_trajectory_refinement`:

- `rigid_correction_xyz_m`: `[dx, dy, dz]` added to the root and every joint.
- `correction_convention`: literal description of that additive convention.
- `correction_magnitude_m`: Euclidean norm of the rigid correction.
- `refined_transl_world`: repeated refined root for consumers that do not read
  the skeleton-shaped field.
- `refined_foot_positions.left|right`: refined low-sole-band XYZ centroid.
- `covariance_m2`: 3x3 diagonal covariance. XY is a bounded inverse local
  robust-factor-precision approximation; Z uses the bounded soft-plane/body
  precision. It is uncertainty metadata, not independent accuracy truth.
- `provenance.plant_anchored`: whether a frozen accepted plant window contributes.
- `provenance.plant_phases[]`: foot, inclusive frame bounds, phase confidence,
  and the soft phase anchor XY.
- `provenance.evidence.{trk,body,plant,smoothness,court_plane}`: each term's
  nominal and final Huber effective weight for that frame.
- `provenance.z_soft_prior`: whether active, sole Z before/after, finite gain,
  bound state, and `clamped_to_plane=false`.

## `placement_trajectory_refinement`

- `config`: the one global configuration used for every clip/player.
- `summary`: global counts, correction median/p95/max, and covariance summary.
- `players`: per-player frame counts, plant-frame counts, and correction summary.
- `policy`: immutable-input, rigid-only, robust-loss, bounded-correction,
  no-clamp, covariance, and protected-eval declarations.
- `provenance.inputs.{skeleton3d,tracks,foot_contact_phases}`: absolute source
  path and SHA-256 for every immutable input.
- `provenance.code_version`: deterministic lane schema/code identity.
- `provenance.coordinate_space`: explicit `court_Z0` declaration plus typed
  coordinate name.
- `provenance.distortion_state`: not applicable because no image transform is
  performed.
- `provenance.preview_band` and `provenance.VERIFIED`: `true` and `0`.

Raw inputs are observations and remain separate. Global fusion must consume
the correction and covariance as a candidate factor; it must not overwrite or
relabel the original BODY/TRK observations.
