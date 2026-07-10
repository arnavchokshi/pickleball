# NS-01.6 cold BODY frame diagnosis

## Result

The failure reproduces locally on the same 45-second `zwCtH` harvest excerpt, but the
introducing commit is **outside** the supplied `460992ae9..d47b399a1` window. Per the lane's
kill rule, this lane stops after diagnosis and does not make a speculative production fix.

## Exact local repro

```bash
PYTHONPATH=. MPLBACKEND=Agg .venv/bin/python runs/lanes/ns016_bodyframes_20260710/repro_real_clip.py
```

The source is `runs/lanes/demo_beststack_gpu_20260710/zwcth45s_demo.mp4`, MD5
`059e396317071e58478e75c55947fe6d`, cut from
`data/online_harvest_20260706/rallies/zwCtH_i1_S4/zwCtH_i1_S4_rally_0001.mp4`.
The fresh run materialized 1,200 JPEGs. BODY requested 200 frames, 18 of which were absent;
the first real lookup failed with the exact production signature:

```text
FileNotFoundError: missing BODY frame image for frame 75; expected body_frames/frame_000075.jpg under /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ns016_bodyframes_20260710/pre_fix_cold_zwcth45s_stride1 or RACKETSPORT_BODY_FRAMES
```

Full output is in `pre_fix_repro_attempt2.log`.

## Root cause

Commit `b437b411886d7fe858e96136ce0d75fb46e95d32` (`checkpoint: SAM-3D-only skeleton
pipeline — RTMW removed, all A100 gates passing`, 2026-07-03) introduced both the frames stage
and `threed/racketsport/process_video_body_frames.py`. Its new schedule unconditionally imposed
`DEFAULT_MAX_SCHEDULED_FRAMES = 1200` and uniformly sampled the materialization union when the
tracked set exceeded that size. BODY independently builds its own bounded compute schedule later.
Nothing guaranteed that BODY's selected frames survived the frames stage's uniform sample.

For this real clip, the frames stage retained 1,200 of 1,315 tracked frames while BODY selected
200 frames; 18 BODY frames were dropped by the independent cap:

`[75, 121, 246, 326, 372, 418, 543, 623, 669, 715, 761, 841, 966, 1012, 1058, 1138, 1263, 1309]`.

The regression endpoints both already contain the defective cap. No scheduling/materialization
file changed in `460992ae9..d47b399a1`; the only owned-file changes there wire array-native BODY
payload assembly and mesh-topology interning after BODY input lookup. Those changes cannot create
or remove JPEGs and do not alter the frames schedule.

## Why Wolverine survives and warm directories can mask the defect

The fresh Wolverine evidence has 244 distinct tracked frames, a 244-frame materialization
schedule, and 244 BODY-requested frames. The 1,200-frame cap never activates, so equality holds.
There is no eval-clip exception in the frames scheduler.

Warm directories can hide the defect because `process_video.py::_stage_frames` skips extraction
when it sees any existing `body_frames/frame_*.jpg` and does not validate the cached set against
the current BODY request schedule. That is a second validation gap, but it is not needed to explain
fresh Wolverine: Wolverine is simply below the defective cap.

The cited rev-9 `w7_critique` artifact is not evidence of successful BODY processing for this
harvest clip: its banked `PIPELINE_SUMMARY.json` shows calibration failed before tracking/frames/BODY.

## Best-stack delta

**(c) NO stack delta.** No production code or manifest was changed because the kill rule fired.

