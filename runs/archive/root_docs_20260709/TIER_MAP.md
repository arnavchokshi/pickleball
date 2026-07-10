# Tier Map

This is a quick mirror of `CAPABILITIES.md` and is not canonical. If this file
conflicts with `CAPABILITIES.md`, fix this file.

| Stage/feature | Tier L0-L3 | Current status | Main artifact |
|---|---|---|---|
| capture guidance | L0/L1 | computed in code; Record UI still needs wiring; Swift/Python sidecar contract and production upload call are P0 blockers | `LiveGuidanceEvaluator`, `PostStopPreviewSummary` |
| player detect/track + foot rings | L0/L1 advisory; L3 authority | live person overlay wired; model not bundled; screen-space today | `LiveFrameTap`, `LiveCourtOverlayEngine`, `tracks.json` |
| live court geometry | L0/L1 next; L2/L3 server paths | no live homography yet; manual/metric/profile server paths exist | `ManualCourtTaps`, `AssistedCourtSeed`, `court_calibration.json` |
| court-plane dot map | L0/L1 after live court lock; L2/L3 today | current live map is proxy; server placement/world can emit real court map | `court_dot_map`, `virtual_world.json` |
| kitchen proximity | L0/L1 after PL-1; L2 after PL-6; L3 review-flag | advisory proximity only live; no live fault verdict | `decide_court_boundary`, `CallsArtifact` |
| serve foot-fault advisory | L0 R&D after PL-1; L1 freeze-frame; L2/L3 after contact wiring | advisory ceiling on single far phone; no live officiating claim | foot rings, contact events, court geometry |
| ball trail + rally segmentation | L0/L1 after PL-5; L2/L3 today | CoreML deploy path proven but student untrained/kill-switched; server WASB path exists | `LiveBallOverlayTracker`, `ball_track.json`, rally artifacts |
| ball in/out advisory | L0 later; L1 challenge replay; L2 fast call; L3 refined call | needs PL-1 + PL-5 for live; server call artifacts are library/orchestrator work | `ball_line_calls`, `ball_inout_uncertainty`, arc artifacts |
| two-bounce / double-bounce | L0 later; L1 after PL-5; L2 after PL-6; L3 library exists | state-machine logic exists in library form, not an authority path yet | `shot_taxonomy`, `excess_bounce` |
| score / serve-side tracking | L0/L1 manual v0; L3 later inference | no CV score tracker today | future H26/P6 score state |
| ball speed | L1 rough estimate after PL-1/PL-5; L2/L3 arcs | no live speed promise before soak benchmark and geometry | solved arc artifacts |
| shot types | L2 after PL-6; L3 P6-1 | `shot_rules_v0.json` landed but unwired | `shot_rules_v0.json`, shot taxonomy outputs |
| highlights / instant replay clips | L0 rally-end trigger; L1 last-capture replay; L3 authority | cheap trigger/replay path, contact-density selection later | rally gating, replay manifests |
| stats + coaching card | L2 basic stats; L3 coaching | placement/court stats are wired but run after the manifest; rally/shot/coaching facts are not production stages | `match_stats.json`, metrics/report/coaching artifacts |
| 3D world / mesh replay | L3 only | never live; remote-GPU BODY remains the offline accuracy path; no full current bundle is verified | `virtual_world.json`, `body_mesh_index/`, `replay_viewer_manifest.json` |
| NVZ momentum / partner-contact | L3 review-flag only | no fixed time-window verdict; human review required | rule flags, world/contact artifacts |

Borderline:

- Camera-space mesh preview is `server-fast`, not phone-real-time.
- LiDAR is a near-field bonus only.
- L0/L1 calls are advisory and uncertainty-banded; L2 is trust-banded and not promotion-grade.
- Preview/scoped/internal-val results are not `VERIFIED`.
