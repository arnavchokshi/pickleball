# Track H visual evidence pack (owner-facing), 2026-07-16

Deliverable: /Users/arnavchokshi/Desktop/visual_evidence_20260716/index.html
(16 assets under assets/; all rendered CPU-only from existing committed artifacts;
no new inference; no fabricated frames; honesty banner: internal cards, preview
band, VERIFIED=0).

Sections and sources:
1. People detection — trk_detbench_20260716/vm_pull (scored tracks + raw dets +
   s06 scorer) over eval_clips mp4s: burlington frame 510 side-by-side (3/4 vs
   4/4), 220-frame side-by-side clip, per-frame coverage timeline (recount matched
   scorer exactly: 427 vs 598 of 600), wolverine weakness frame (5 confident
   non-player FPs red + drifted 4th track amber; 16 true spectator FPs, cov4 0.727).
   FP labels grounded by IoU vs scored tracks, NOT a size heuristic (first attempt
   mislabeled a far-side player; caught in manager eyeball pass and fixed).
2. Foot placement — trackI_placefuse_20260716: summary bars 34.6/33.6/20.8/48.4 ->
   6.7/5.6/6.3/6.8 mm (exact scored values), 4 per-clip worst-plant-phase zooms
   (baseline reconstructed per frame as fused - rigid_correction; baseline
   recomputation matches scored slide_m to ~1e-13 on 3/4 clips; curve labels state
   the wander of exactly the drawn window).
3. Ball arc — ballarc_anchorfusion_20260716 + ballarc_scale_guard_20260715 +
   salvaged pbvision ball_track.json: 3-row timeline (12789-visible-frame 2D track,
   baseline 1/188 fitted, balanced 85/361 fitted with 18 red-hatched flight-sanity
   windows) + kill-reasons panel (court volume +78.5 m, speed jump 30.8 vs 8.75,
   heading 164.6 deg, multi-apex; balanced+conservative KILLED, broad crashed;
   baseline in-rally coverage 0.27% -> 29.65%; asserted against artifacts in-script).
4. 3D world live — viewer at http://127.0.0.1:5199/?manifest=... : server was
   running WITHOUT VITE_REPLAY_VERIFY_DEV_BYPASS=1 (landed on sign-in); restarted
   nohup-detached with the flag (see vite5199.log), re-verified opens straight to
   replay (video readyState 4, canvas, strip, 9 badges, zero errors); fresh
   screenshot + webux3 before/afters embedded.

Browser verification (verify_pack.py, headed Chromium): 15/15 images loaded,
mp4 plays (readyState 4), viewer link live (200), zero page errors -> PASS.
Page screenshots: pack_top.png / pack_bottom.png.

Queued next (coordinator): OneWorldV1 fused-world rendering in the viewer —
gated on Track K schema commit; do not start before the gate opens.
