# py-spy stall evidence — ball_arc segment-association scaling defect (2026-07-16T02:35Z)

Captured live by the Track A manager on pickleball-h100-pbv11r while stage `ball_arc` had produced
no artifact for ~2h50m (CPU ~123%, GPU 0%). The Sonnet driver lane took two additional samples 15+
minutes apart at the IDENTICAL location on `segment_id: 7` (~5.16s gap integrated at 1/240s across
a large unassigned candidate pool, ~1240 RK4 substeps per `predict()` call). The SIGINT traceback at
teardown (see vm_pull_partial/run_stdout.log, md5 f10a81bcc64f9f3b41dc7c050fd03acc) terminated in
`ball_arc_solver.py:6411 _add_state` — the same RK4 state math. Three independent captures, one
location.

```
Thread 3999 (active+gil): "MainThread"
    drag_k_per_m (threed/racketsport/ball_arc_solver.py:68)
    deriv (threed/racketsport/ball_arc_solver.py:5490)
    _rk4_step (threed/racketsport/ball_arc_solver.py:5494)
    _integrate_positions (threed/racketsport/ball_arc_solver.py:5471)
    predict (threed/racketsport/ball_arc_solver.py:331)
    _select_candidates_for_segment (threed/racketsport/ball_arc_solver.py:1697)
    _fit_flight_segment_with_candidate_association (threed/racketsport/ball_arc_solver.py:1550)
    fit_flight_segment (threed/racketsport/ball_arc_solver.py:476)
    _fit_anchor_pair (threed/racketsport/ball_arc_solver.py:2483)
    _fit_segments_from_anchors (threed/racketsport/ball_arc_solver.py:2440)
    solve_ball_arc_track (threed/racketsport/ball_arc_solver.py:2225)
    solve_arc_with_flight_sanity (threed/racketsport/ball_arc_chain.py:955)
    run_default_ball_arc_chain (threed/racketsport/ball_arc_chain.py:241)
    _stage_ball_arc (process_video.py:2882)
    _run_stage_safely (process_video.py:1154)
    _run_stage_list (process_video.py:940)
    _run_serial (process_video.py:957)
    run (process_video.py:893)
    main (process_video.py:7651)
    <module> (process_video.py:7665)
```

Timing context (artifact mtimes, UTC): pipeline start 23:06; calibration 23:07; ball WASB chain
23:12-23:44 (~37 min for 20,922 frames); ball_arc entered ~23:44 and never completed before the
02:50Z SIGINT (3h06m in-stage). Run was a fresh content-addressed generation at pin ac0b14ab0.
