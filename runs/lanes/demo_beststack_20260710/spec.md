# demo_beststack_20260710 — owner demo: best-stack E2E + fail-closed ball emission

Owner directive (2026-07-10 ~00:4xZ, verbatim intent): back in 4-5 hours; wants a demo video
showing every improvement from this week that is actually good, produced by the best pipeline
we have, everything spun up and put together the best way possible. Standing directive going
forward: promote better components into the default E2E when they win their gates.

## Scope

1. Fail-closed 3D ball world emission (SHIP-FIRST item from runs/research_ball3d_20260709/RULINGS.md,
   single-fix boundary from runs/lanes/w7_ball3ddiag_20260709/DIAGNOSIS.md):
   a segment with fallback fit status AND (failed flight sanity OR insufficient inliers OR
   visible reprojection p95 > 18px) contributes world_xyz=null; arc band/demotion/provenance
   propagates to confidence gating instead of being relabeled `measured`.
   Recorded as a best_stack.json default entry with provenance (default selection, NOT a
   VERIFIED claim; VERIFIED=0 binding).
2. One H100 spot (w7close snapshot), self-torn: cold fresh-clip-dir best-stack E2E on
   wolverine_mixed_0200_mid_steep_corner + owner critique clip; save `best_stack.resolved`
   attestation from PIPELINE_SUMMARY.json; pull worlds + md5.
3. Demo video (before/after per category: court, tracking, ball 2D, ball 3D old-vs-new,
   BODY mesh + smoothing + ghost + paddle, stats, speed) delivered to ~/Desktop.

## Fences

- Working in worktree demo-beststack-20260709 branched at committed HEAD d47b399a1.
- NOT touching: ns014 decode files (mhr_decode/hmr_deep/coordinates/gates), court lane files,
  calibrate_charuco_device.py, or the identity session's uncommitted process_video.py hunks.
- No protected Outdoor/Indoor eval labels read or written. No promotion claims; demo video is
  explicitly labeled where a component is estimated/preview.

## Exit

Video on Desktop + lane REPORT.md with attestation + branch pushed + draft PR.
