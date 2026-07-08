# Playback Decision Table

Lane: `w6_playbackdiag_20260708`  
Mode: read-only diagnosis. I wrote only this lane directory. No network, no GPU, and no protected eval labels were read.

## Evidence

- Wave-6 owner critique: `runs/manager/wave6_boot_prompt.md`
- Close-proof worlds: `runs/lanes/w5_closeproof_20260708/{burlington,wolverine,outdoor,img1605_rerun2}/`
- Viewer source: `web/replay/src/App.tsx`, `web/replay/src/viewerData.ts`
- P2-2 evidence: `runs/lanes/w5_p22latent_20260707/`, `runs/lanes/w5_p22wiring_20260708/`

## Current Mesh Density

`frame_compute_plan.summary.world_mesh_frame_count` is 100 for all four close-proof runs. `body_mesh_index.summary.mesh_frame_count` is player-frame count, not unique source-frame count.

| clip | fps | total frames | BODY scheduled frames | mesh selected source frames | mesh player-frames in index | effective mesh fps at 1x | worst mesh gap |
|---|---:|---:|---:|---:|---:|---:|---|
| Burlington | 59.94 | 600 | 675 | 100 | 166 | 9.99 | 83 frames, about 1.38s |
| Wolverine | 30.00 | 300 | 266 | 100 | 276 | 10.00 | 21 frames, about 0.70s |
| Outdoor | 60.00 | 1151 | 1251 | 100 | 169 | 5.21 | 110 frames, about 1.83s |
| IMG1605 rerun2 | 29.99 | 297 | 243 | 100 | 148 | 10.10 | 56 frames, about 1.87s |

## BODY Scheduling Density

BODY scheduling is not the main source of stutter in Burlington/Wolverine/Outdoor: the unique scheduled BODY frames are contiguous at source-frame cadence. IMG1605 has one real BODY schedule hole.

| clip | BODY gap histogram, frames | max BODY gap | median BODY gap | likely worst visible stutter |
|---|---|---:|---:|---|
| Burlington | `1:599` | 1 | 1 | mesh cap/index gap, especially frame 343 -> 426 |
| Wolverine | `1:243` | 1 | 1 | mesh cap/index gap, especially frame 270 -> 291 |
| Outdoor | `1:1150` | 1 | 1 | mesh cap/index gap, especially frame 483 -> 593 |
| IMG1605 rerun2 | `1:241, 55:1` | 55 | 1 | combined BODY and mesh gap, frame 82 -> 137 / 81 -> 137 |

## Mesh Economy

The close-proof runs did not build/fetch `body_mesh.json` or `smpl_motion.json`; each `PIPELINE_SUMMARY.json` says monoliths were not built in speed-default mode. The table below uses actual recursive `body_mesh_index/` bytes and linear extrapolation from measured per-frame variable bytes plus one shared `body_mesh_faces.json`.

| clip | actual index MiB at 100 selected frames | cap 200 MiB | cap 400/all MiB | all scheduled frame count | measured monoliths available locally |
|---|---:|---:|---:|---:|---|
| Burlington | 13.57 actual; 45.59 linear normalized | 89.45 | 177.17 / 297.79 | 675 | older Burlington `body_mesh.json` 65.1 MB |
| Wolverine | 21.31 actual; 30.11 linear normalized | 58.49 | 77.22 / 77.22 | 266 | Wolverine `body_mesh.json` 304.7 MB and older 447.3 MB |
| Outdoor | 13.68 actual; 151.16 linear normalized | 300.58 | 599.43 / 1871.04 | 1251 | none found locally |
| IMG1605 rerun2 | 12.26 actual | 22.79 | 27.32 / 27.32 | 243 | none found locally |

Interpretation: the 100-frame cap is the dominant visible mesh cadence limiter. Raising the cap is roughly linear in transferred/viewer-loaded mesh bytes, but exact slope varies by chunk packing and number of players per selected frame.

## Viewer Capability Today

`web/replay/src/App.tsx` exposes `DisplayFpsControl` as `2x FPS (interpolated)`.

`web/replay/src/viewerData.ts` does two display-only things:

- `doubleFpsWorld(...)` inserts midpoint world/player frames by linearly interpolating `joints_world`, `joint_conf`, `track_world_xy`, `transl_world`, `floor_world_xyz`, and mesh-ref metadata.
- `doubleFpsBodyMesh(...)` inserts midpoint mesh frames only when adjacent mesh frames are in the same `source_window_index` and have matching vertex/joint counts. The synthetic frame has `mesh_interpolated=true` and `interpolation` metadata.

The viewer already has vertex correspondence for loaded mesh frames because all frames use the same topology and `body_mesh_faces.json`. It also has per-frame joints. It does not have MHR latent pose codes or a decoder in the viewer bundle, so principled latent inter-frame BODY interpolation is not available today. Render-side interpolation must stay cosmetic and visibly labeled; it must never be exported as measured evidence.

## GPU Cost

Measured H100 steady inference from `body_stage_phase_timing.json` is about 15.2 ms per person-frame. This is only the steady SAM3D bucket inference component; full BODY stage wall includes preprocessing, model setup, gates, index build, transfer, and other overhead.

| clip | person-frames | BODY stage wall | steady ms/person-frame | +2x steady GPU min estimate | +4x steady GPU min estimate |
|---|---:|---:|---:|---:|---:|
| Burlington | 1785 | 561.536s | 15.233 | +0.45 | +1.36 |
| Wolverine | 680 | 379.912s | 15.278 | +0.17 | +0.52 |
| Outdoor | 2653 | 565.240s | 15.178 | +0.67 | +2.01 |
| IMG1605 rerun2 | 359 | 261.447s | 15.419 | +0.09 | +0.28 |

Denser scheduling alone does not fix the 100-frame mesh cap. It only helps if more frames become eligible for mesh export and the cap/index policy is also raised.

## P2-2 Tie-In

P2-2 phase 1 exists as an MHR decode wrapper and latent smoother prototype, but it is not wired into `process_video.py` or the viewer. `runs/lanes/w5_p22wiring_20260708/report.json` is `PARTIAL`: decoded smoothed candidate dirs were absent, local decode was blocked by missing `roma`, and the Wolverine table is explicitly `world_joint_proxy_not_latent_decode`.

What P2-2 could enable after wiring: interpolate/smooth latent BODY pose codes, decode them, and keep mesh+skeleton coherent by construction. What exists today: proxy evidence only, no viewer-bundled latents, no wiring-ready lambda, and no measured playback-fps gain.

## Mechanism Table

| mechanism | measured current state | quantified cost | expected playback-fps gain | NEVER-measured-evidence risk | composes with |
|---|---|---|---|---|---|
| render-interp | Existing 2x display control interpolates skeletons and eligible same-window meshes; cosmetic only. | 0 disk, 0 transfer, 0 GPU; viewer-only engineering. | Skeleton display can double. Mesh gain is bounded by available mesh frames; long inter-window gaps still need explicit cosmetic blending. | High unless labeled. Use "Display FPS: interpolated/cosmetic"; tick measured frames separately; disable metric/export use. | all options |
| mesh-cap-raise | All clips selected 100 mesh source frames; Outdoor is only 5.21 mesh fps. | Linear disk/transfer/viewer load; see cap table above. No extra inference if exporting already scheduled BODY frames. | Cap 200 roughly doubles mesh cadence where scheduled frames exist; 400/all can approach 20fps+ for most clips. | Low: emitted frames are measured BODY outputs, still scoped/unverified. Label cap/source count. | render-interp, denser scheduling, P2-2 |
| denser-scheduling | BODY schedule is already source-contiguous for 3/4 clips; IMG1605 has one 55-frame hole. | Steady H100 inference estimates above; full wall scaling larger. Scheduler/guard engineering. | Does not fix mesh cap by itself; helps only if BODY is sparse or cap/index policy also raises. | Low for emitted BODY frames; cost/quality drift risk higher. Preserve schedule provenance. | mesh-cap-raise, render-interp, P2-2 |
| P2-2-latent-interp | Prototype/unwired; current evidence proxy-only and not wiring-ready. | Unknown disk/GPU until latent payload/decode wiring; engineering scope is raw-latent persistence, strict gate, decoded candidates, viewer/pipeline decode, trust-band UI. | Potential coherent inter-frame mesh+skeleton once wired; no measured gain today. | High unless visibly banded as latent-interpolated and excluded from metrics/training/promotion. | all options |

## Proposed BUILD_CHECKLIST Bullet

`[W6 PLAYBACKDIAG 2026-07-08, Codex READ-ONLY] Measured close-proof 3D playback cadence and cost menu in runs/lanes/w6_playbackdiag_20260708/: all 4 close-proof runs selected 100 world-mesh source frames; effective mesh cadence is Burlington 9.99fps, Wolverine 10.00fps, Outdoor 5.21fps, IMG1605 10.10fps with worst mesh gaps 83/21/110/56 frames. Viewer 2x FPS is cosmetic render-side interpolation only; mesh cap raise is linear storage/transfer; denser scheduling has measured H100 steady cost but does not fix the cap alone; P2-2 latent interpolation remains unwired/proxy-only.`

## Honest Issues

- Close-proof monolith sizes could not be measured for those exact four runs because the speed-default BODY path did not build/fetch `body_mesh.json` or `smpl_motion.json`.
- Older local monoliths exist for Burlington/Wolverine only; they are not the exact close-proof monoliths.
- GPU cost is split into measured steady inference estimates and measured full stage wall; I did not extrapolate full wall as if every overhead scaled linearly.
- Viewer behavior was grep/read verified only. I did not run web tests or a browser because this lane is read-only and the deliverable is diagnosis.
