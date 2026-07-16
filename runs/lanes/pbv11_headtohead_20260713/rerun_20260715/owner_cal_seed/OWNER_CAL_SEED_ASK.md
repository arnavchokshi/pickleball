# Owner ask (2 minutes): trusted 4-corner court seed for the pb.vision demo video

STAGED 2026-07-15 — NOT yet in OWNER_CHECKIN.md (waiting on Track C's window-close commit per
coordinator; the Track A manager will add it there afterward via the inflight ledger).

WHY: both MOVE-1 head-to-head attempts ran with `--allow-auto-court-corners-preview` because no
trusted calibration seed exists for this video. That seed graded **poor** and fail-closed all
metric world output (`ball_world`, `virtual_world_metric`) — so even a completed run cannot produce
the court-frame 3D comparison vs pb.vision. Your 4 corner taps make the next run's calibration a
`corrected_unverified` human seed (the same class our eval clips use), unlocking metric output.

## The 2-minute flow

1. Open `tap_corners.html` (this folder) in any browser — it shows the chosen frame
   (`candidate_t10s_frame300.jpg`, t=10s: all four corners visible, no occlusion).
2. Click the 4 outer court corners in the order it prompts:
   far_left → far_right → near_left → near_right. Zoom (⌘+) first if you like; press R to redo.
3. Copy the JSON it produces back to the manager (or save it as `owner_court_corners.json` in this
   folder). Done.

Alternates if the primary frame looks wrong to you: `candidate_t60s_frame1800.jpg`,
`candidate_t300s_frame9000.jpg` (edit the `src` line in the HTML, note which one you used).

## What happens next (engineering, not you)

The taps are wrapped into the runner's `court_corners.json` seed format
(`annotation.items[0].court_corners` + `image_size [1280,720]`, `source: human_review`,
`status: corrected_unverified` — same shape as
`eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_corners.json`), validated by
running the calibration stage locally, and recorded as the trusted seed for any third MOVE-1
attempt (which remains gated on the coordinator's explicit GO + the ballarc guard lane +
harness v2 green). This seed is `corrected_unverified` provenance — it enables metric output; it
is NOT a CAL accuracy promotion. VERIFIED=0 stands.
