# T1 gold capture — executable shot list (one visit, 60–90 min)

Purpose: Tier-1 of the ball-3D plan — **100–300 controlled flights** to validate sync, calibration,
triangulation, coordinates, and metrics (`runs/ball3d_lifting_plan_20260723/PLAN.md`, tier table +
artifact A-1). Only you can produce this (PLAN.md: "ONLY the owner can produce the gold multi-view
set"). Companion background: `runs/manager/capture_recording_guide_20260709.md`.

## Equipment (bring all of it)
- **3 phones**: your production baseline iPhone + 2 temporary phones able to record **120–240 fps**
  (PLAN A1). Charge full; power banks. **≥10 GB free each** (1080p60 HEVC ≈ 4–6 GB/h, guide §C).
- 3 tripods/mounts (two must get the camera **elevated**, 2–4 m if possible — fence, bleacher, pole).
- Printed **ChArUco/checkerboard** on rigid backing (intrinsics, PLAN A-2).
- Tape measure (≥10 m), painter's tape or chalk (bounce markers), pen + printed log sheet (below).
- Ball machine if available, else a consistent feeder partner. **≥6 identical balls** — write down
  brand/model (ball radius enters the GT convention, PLAN A3).

## Camera placement (PLAN A-1: one elevated side, one elevated opposite corner/far side)
```
                      far baseline
        +--------------------------------+
        |            FAR half            |     CAM-CORNER: elevated, outside the
        |   [F1]      [F2]      [F3]     |  <- far corner OPPOSITE CAM-SIDE's side
        +---------------- net -----------+
        |   [N1]      [N2]      [N3]     |
        |           NEAR half            |  <- CAM-SIDE: elevated at mid-court
        +--------------------------------+     sideline (near the net post)
                     near baseline
                          ^
              CAM-BASE: production iPhone, elevated,
              centered behind the near baseline
              (usual app framing: full court + both baselines)
```
Settings: all landscape. CAM-BASE = your normal app mode (1080p60; app enforces landscape + audio,
guide §A). CAM-SIDE / CAM-CORNER = highest fps available (120 or 240), lock focus/exposure if the
phone allows, **note the stabilization mode and never zoom after setup** (stabilization/crop changes
effective FOV — PLAN A-2). After calibration, nobody touches any camera. If one gets bumped: log it,
re-show the board to that camera, do a fresh sync event.

## Sync method (PLAN A-1: audio + visual, MULTIPLE events — never one clap)
A "sync event" = stand where all 3 cameras see you, then **3 sharp claps + a flashlight blink**
(phone torch on/off) per event. Do one at start, one between every block, one at the end (≥5 total).
Cameras record **continuously** from the first sync event to the last — do not stop between flights.
If any camera must restart, do an immediate extra sync event.

## Court prep (known bounce points)
Tape 6 markers and measure each with the tape measure from the two nearest lines (write the offsets
on the log sheet): N1/F1 = kitchen (NVZ) center; N2/F2 = service-box center; N3/F3 = 1 m inside the
baseline, centered. Also measure **net height at both posts and center** (PLAN A3 net-top curve).
Photograph every marker with the tape measure in frame, plus each camera position.

## Flight matrix (core = 160; anything ≥100 with ≥20 per family is a success)
| Block | Family | From | Target markers | Count |
|---|---|---|---|---|
| 1 | Serve | near baseline → far, then swap | F2 then N2 (service boxes) | 20 + 20 |
| 2 | Drive | baseline, flat/hard | F3 then N3 (deep) | 20 + 20 |
| 3 | Dink | at NVZ line, soft | F1 then N1 (kitchen) | 20 + 20 |
| 4 | Lob | mid-court, high apex | F3 then N3 (deep) | 20 + 20 |

Rules: unobstructed single flights (no rallies, no bodies between ball and cameras); let the ball
bounce and roll out before the next feed; aim at the marker — a miss is fine, just log where it
actually bounced. Near AND far halves are deliberate coverage (PLAN A-1 / T1 row).

## Per-flight log sheet columns
`flight_id | block/family | feed (machine/hand) | feeder position | target marker | actual bounce
vs marker (on / short / long / left / right, est. cm) | clean flight? (y/n) | redo? | notes`
Header once per session: date, venue, indoor/outdoor, lighting, surface, ball brand/model, net
heights (3), marker offsets (6), per-camera phone model + video mode + height/position + stabilization.

## Timeline (75 min nominal)
- 0–10 set up 3 cameras, frame, photo each rig
- 10–20 calibration: ChArUco shown to each camera (~30 s each, tilt it around); net measured;
  markers taped, measured, photographed
- 20–22 SYNC 1 → 22–32 Block 1 (serves) → 32–34 SYNC 2 → 34–44 Block 2 (drives) → 44–46 SYNC 3
  → 46–56 Block 3 (dinks) → 56–58 SYNC 4 → 58–68 Block 4 (lobs) → 68–70 FINAL SYNC 5
- 70–80 stop recordings, spot-check each file plays, pack up
- 80–90 transfer starts (below)

## File naming + delivery
Copy **originals** (cable/Image Capture or AirDrop set to keep original quality — never a messaging
app that recompresses). Rename only after copying, never re-encode:
`t1_<yyyymmdd>_<venue>_cam{base|side|corner}_seg<NN>.mov`
Deliver one folder to the Mac: `data/owner_t1_capture_<yyyymmdd>/` containing `cam_base/`,
`cam_side/`, `cam_corner/`, `photos/` (rigs, markers+tape, board, net measurements), and
`log_sheet` (photo or scan). The controller registers it in the data ledger — you're done.

## Done-when checklist
- [ ] 3 cameras, full court + net visible in each, continuous recordings
- [ ] ≥5 sync events (clap + flash), one between every block
- [ ] Board shown to each camera; net measured at 3 points; 6 markers measured + photographed
- [ ] ≥100 logged flights, ≥20 per family, both halves covered
- [ ] Originals + photos + log sheet delivered as one folder
