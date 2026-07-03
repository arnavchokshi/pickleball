# Tier Map

This is a quick mirror of `CAPABILITIES.md` and is not canonical. If this file
conflicts with `CAPABILITIES.md`, fix this file.

| Stage | Tier | Current status | Main artifact |
|---|---|---|---|
| capture/setup guidance | on-device live | scoped iOS scaffold/pass slices | `capture_sidecar.json` |
| calibration seed | on-device setup plus server refine | scaffold/preview | `court_calibration.json` |
| person/pose preview | on-device live | scaffold/scoped | `tracks.json`, pose priors |
| ball/contact preview | on-device live | scaffold | `ball_track.json`, `contact_windows.json` |
| tracking authority | server offline | in progress | `tracks.json` |
| BODY mesh | server offline | scaffold | `smpl_motion.json`, `body_mesh.json` |
| foot/physics | server offline | internal-val done only | `physics_*.json`, `virtual_world.json` |
| paddle 6DoF | server offline | scaffold | `racket_pose.json` |
| metrics/report | server offline | scaffold | metrics/report artifacts |
| replay/world | server offline plus native/web playback | scoped pass | `virtual_world.json`, `replay_viewer_manifest.json` |

Borderline:

- Camera-space mesh preview is `server-fast`, not phone-real-time.
- LiDAR is a near-field bonus only.
- Preview/scoped/internal-val results are not `VERIFIED`.
