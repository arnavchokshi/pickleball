# Integrated end-to-end demo clips — court + players + skeletons + meshes + ball + events

Lane: `integrated_demo_20260729`. Date: 2026-07-29. Commit `285f5e7`.
Status: rendered product demo, preview-band; `VERIFIED=0`. Written by the
orchestrator from the lane's returned findings. Delivery:
`~/Desktop/visual_evidence_20260728/integrated/` (3 MP4s + index.html +
INTEGRATED_RESULTS.md + render_manifest.json); renderer committed at
`render_overlay.py`.

## Runs (full preset, co-located `--body-local --one-world`, night1 A100)

| Clip | Wall (s) | Status | Typed degrades | Frames |
|---|---:|---|---|---:|
| Wolverine (reused stretch demo) | 987.3 | partial | input_quality, ball_arc, grounding_refine, ball_arc_refined, coaching_facts | 300 @30 |
| Burlington (fresh) | 1531.6 | partial | input_quality, ball_arc, ball_arc_refined, coaching_facts | 600 @60 |
| Outdoor webcam (fresh) | 2370.2 | partial | input_quality, ball_arc, ball_arc_refined, coaching_facts | 1151 @60 |

All degrades are typed and documented product behaviors (segment-budget kills,
zero-fabrication coaching rejection); nothing fabricated.

## Coverage in the rendered clips

| Clip | Ball confident/weak/not-rendered % | Contact events | Mesh player-frames (windows) |
|---|---|---:|---:|
| Wolverine | 14.0 / 37.3 / 48.7 | 23 | 1164 (7) |
| Burlington | 21.3 / 5.3 / 73.3 | 27 | 2073 (4) |
| Outdoor webcam | 7.9 / 2.8 / 89.3 | 32 | 3448 (33) |

Bands from `ball_track_arc_solved.json`: anchored_measured + arc_interpolated
rendered "confident," arc_weak rendered visually distinct "weak," hidden (and
15 arc_extrapolated frames on outdoor) NOT rendered as measurement — a
mislabeling bug (percentages summing to 98.7%) was caught, fixed, re-verified
to 100%. Hidden-band frames confirmed to draw no marker in the composite
logic, not just the report.

## Verification

ffprobe frame-count exact vs source for all three; multiple frames per clip
extracted and visually inspected (court lines/NVZ shading, skeletons, violet
mesh rings, CONTACT banners, minimap with player+ball dots, trust legend).

## Honest notes

The renders show the ball program's current truth: confident-band coverage is
14-21% on the best clips — the trained-event/arc work (E-v2 + kink corpus, in
flight) is what raises it. Mid-lane, a sibling commit advanced shared HEAD;
handled by re-syncing the VM before the outdoor dispatch (burlington ran
against a strict ancestor, unaffected).
