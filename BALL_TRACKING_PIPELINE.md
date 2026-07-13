# Ball Tracking Pipeline

Last updated: 2026-07-13.

This is the focused ball-stage contract. It exists because ball code comments
refer to section numbers here. The global product plan and current BALL gate
live in `NORTH_STAR_ROADMAP.md`.

Current truth: BALL is not verified. The pipeline can run/reuse ball artifacts
and render confidence-banded preview output, but no reviewed BALL acceptance gate
has passed.

## 1. Scope

Ball covers:

- 2D image track.
- Bounce/contact/in-out events.
- Audio and wrist/pose cue fusion.
- Optional physics/world uplift for preview and replay.
- Confidence/uncertainty labels.

It does not own person tracking, BODY, paddle pose, or coaching copy.

## 2. Live vs Offline

| Tier | Purpose | Authority |
|---|---|---|
| on-device live | fast ball cue/trail only when confidence is defensible | preview |
| server offline | accurate track/event/in-out processing from the full video and sidecars | future authority after gates pass |

The live tier may upload priors. The server tier must recompute or validate
before producing authoritative metrics.

## 3. Stage Inputs

| Input | Source |
|---|---|
| `ball_track.json` | WASB/TrackNet/candidate tracker or explicit `--ball-track` reuse. |
| `court_calibration.json` | CAL sidecar/metric/manual seed for court-plane reasoning. |
| `audio_onsets.json` | audio event extraction when audio exists. |
| `wrist_velocity_peaks.json` | BODY/pose-derived wrist cue when trustworthy. |
| `events_selected.json` | physically validated contact events from arc solving, when present. |
| `ball_track_arc_solved.json` | arc-solved ball world track, when present. |

## 4. Output Contracts

Here, `measured` names an image-space observation/sample role, not reviewed GT
or promotion; model-produced observations retain model-estimated provenance.
`predicted` is physics-derived/render-only, and `diagnostic` is never authority.

| Artifact | Trust label | Meaning |
|---|---|---|
| `ball_track.json` | measured/model-estimated | Selected image-space samples with visibility, confidence, and source; not reviewed GT. |
| `ball_candidates.json` | measured/model-estimated | Raw top-K detector candidate evidence retained separately from the selected track. |
| `ball_inflections.json` | diagnostic | Trajectory-turn proposals; not contact authority. |
| `audio_onsets.json` | measured/diagnostic | Audio pop/onset observations used only as candidate cues. |
| `contact_windows.json` | diagnostic/model-estimated | Pre-BODY fused candidate contact windows. |
| `contact_windows_refined_v1.json` | diagnostic/model-estimated | Separate post-BODY contact candidates; raw `contact_windows.json` remains immutable. |
| `ball_bounce_candidates.json` | diagnostic | Auto-proposed bounce anchors; not accepted bounce truth. |
| `ball_size_observations.json` | measured/diagnostic | Source-pixel WASB heatmap/blob extents; emission-only, not GT or depth authority. |
| `ball_track_arc_solved.json` | predicted | Physics-predicted 3D arc segments; render-only and self-kill gated. |
| `ball_arc_render.json` | predicted | Dense render samples derived from accepted solved arcs; never measured evidence. |
| `ball_flight_sanity.json` | diagnostic | Flight-sanity demotions and failure reasons; not 3D accuracy proof. |
| `ball_track_physics_filled.json` | predicted | Render-honest filled/derived samples. |
| `events_selected.json` | diagnostic/reviewed when applicable | Manually or physically selected event set when available; provenance controls authority. |

## 5. Runtime Policy

### 5.1 Rally Spans

Process rally spans when possible, with padding around active intervals. Padding
defaults should preserve context around contacts and bounces rather than clipping
the event itself. `rally_gating.py` owns the exact implementation.

### 5.6 Bounce Uncertainty

Every in/out decision must carry uncertainty. If the ball-to-line margin is not
larger than the modeled uncertainty, the output is `too_close_to_call`. Single
camera line calls are never officiating-grade by default.

Uncertainty can include:

- pixel localization error,
- calibration/reprojection error,
- ball radius and contact patch ambiguity,
- timing uncertainty from frame rate/audio alignment,
- track smoothing/physics residuals.

## 6. Constants And Targets

Current code uses these as engineering targets, not proof:

- Bounce timing target: within about 40 ms when audio/review evidence supports it.
- Ball radius uncertainty: about 2 cm.
- Straight-segment jitter target: under about 2 px std.
- Hidden false positives must be tracked separately from visible-ball miss rate.

## 7. Confidence Bands

Ball samples must distinguish:

- measured visible samples,
- hidden/no-prediction samples,
- physics-derived or filled samples,
- low-confidence preview samples,
- accepted/reviewed events when present.

The replay may render low-confidence ball data, but it must badge it honestly.

## 8. Killed Or Rejected Patterns

Do not promote:

- CVAT-clip-only fine-tunes that regress held-out behavior,
- `VNDetectTrajectoriesRequest` rung-1 ball tracking. The July 2026 spike was
  fast but failed the kill gate: best recall@20 stayed under 0.5 on Burlington
  and Wolverine, so it is not a ball-position path. Keep any future on-device
  ball work on the distilled CoreML heatmap path unless new reviewed evidence
  reopens this.
- local-search postprocess variants that reduce recall or increase hidden FP,
- TrackNet/WASB fusion by veto when it collapses visible recall,
- in/out calls without modeled uncertainty and a gray zone.

## 9. Acceptance Gates

Promotion requires reviewed-label evidence, not artifact existence:

| Gate | Requirement |
|---|---|
| M1 track | High ball F1 at pixel tolerance, recall floor, hidden-FP ceiling. |
| contact/bounce | reviewed event timing within target tolerance. |
| in/out | confident calls agree with review; uncertain calls become gray-zone. |
| replay | replay consumes ball samples with correct trust bands. |

BALL is `WIRED_DEFAULT` and partially measured, not VERIFIED; component and
product gates remain open.

## 10. Default 3D Ball Chain (2026-07-05)

`process_video.py` now runs the ball 3D chain BY DEFAULT after the ball stage: top-K candidate
sidecar emission during detector inference -> label-free auto-bounce anchor proposal
(`scripts/racketsport/propose_ball_bounce_candidates.py`) -> event-anchored arc solver at the frozen
validated configuration (`threed/racketsport/ball_arc_chain.py`; config equality is fixture-tested) ->
parabolic flight-sanity demotion (`threed/racketsport/ball_flight_sanity.py`). Outputs are render-only
and fail closed: a self-killed solve (`experimental_off`/`degenerate_zero_segments`) is written to disk
but never consumed by `virtual_world.py`. The web viewer renders the ball with a trail, band-honest
measured/predicted styling, impact markers, and a coverage KPI. Opt-outs: `--no-ball-arc`,
`--no-ball-candidates`. Standalone runner: `scripts/racketsport/run_ball_chain.py` (held-out clips
require `--heldout-authorized`). Verification: `scripts/racketsport/verify_process_video_viewer.py
--screenshot-at-seconds N` captures mid-playback proof.
