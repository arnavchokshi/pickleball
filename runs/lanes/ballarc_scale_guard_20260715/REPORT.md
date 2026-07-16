# Ball arc production-scale guard report

Date: 2026-07-16  
Lane: `ballarc_scale_guard_20260715`  
Disposition: correctness/scaling safety fix only; `VERIFIED=0` remains binding; no promotion.

## Acceptance result

The CPU-only default ball arc chain completed on the salvaged 20,922-frame, approximately 697 s
input in **1,493.096 s (24.885 min)** with exit code 0 and zero stderr bytes. This is below the
coordinator's 1,800 s target. The solver did not manufacture arcs when the numerical work exceeded
its budget: the solved artifact and chain manifest both report `status=degraded`, 188 emitted
segments, 187 explicit missing segments, and
`missing_segment_reasons={"segment_budget_exceeded": 187}`. Every one of those 187 segments has a
typed degradation payload with `outcome_type=segment_budget_exceeded`, `reason` of the same value,
`evidence_provenance=missing`, and `authority=degraded`; the audit found zero malformed timeout
payloads. One segment fit. The result is bounded abstention, not an accuracy improvement.

The optional legacy `ball_physics3d` diagnostic reference is explicitly reported as
`not_run_due_to_segment_budget_exceeded` after production arc degradation. Running that separate,
full-game bounce-window optimizer after a typed production abstention would defeat the end-to-end
bound. Its missing/degraded authority and reason are present in the solved artifact. Its existing
path remains unchanged when no production segment budget trips.

## Reproduction and diagnosis

The salvaged R&D-reference input contains 20,922 frames, 12,789 visible primary observations,
12,837 nonempty candidate frames, 35,345 sidecar candidates, and 20 automatic bounce candidates.
These are comparison inputs only: they are not ground truth and are not training data.

The old-equivalent local run produced no completed artifact after 329.9 s and was interrupted while
executing RK4 work. The independent VM evidence is stronger: the production stage remained in
`ball_arc` for 3 h 06 min, with three samples at
`_select_candidates_for_segment -> predict -> _integrate_positions -> _rk4_step` on segment 7.
That VM segmentation had an approximately 5.16 s gap, so each prediction integrated approximately
1,240 RK4 substeps at the fixed 1/240 s step.

The current local tree selected a wider final segment 7 from frames 7,951 through 11,091:
104.666667 s, 2,191 primary observations, 2,198 candidate frames, 8,381 candidates, and 25,120
full-gap RK4 substeps per prediction. Across the final candidate pass, the maxima were the same
8,381 candidates, 2,198 candidate frames, and 25,120 full-gap substeps.

The pool enters association because `_candidate_sets_in_span` accepts every visible candidate in
the anchor time interval and limits only each frame to 12 candidates; it has no span-duration or
total-pool cap. `_select_candidates_for_segment` then calls `segment.predict(candidate.t)` for each
candidate, and every prediction integrates forward again from the segment start. This repeats for
up to five association iterations plus numerical refits. The dominant work is therefore
approximately:

`candidate count x RK4 substeps from segment start x association/refit iterations`.

The checked-in real-artifact regression uses the maximum-density 156-frame window from frames
7,940 through 8,095. The original window is 5.166666 s with 484 real candidates. The CI trim keeps
every fourth real frame plus the final endpoint, rebases only frame indices/timestamps, and retains
all candidates on those frames, leaving 121 real candidates. It asserts the segment is still
present, times out within its small test budget, and appears at segment, artifact, and summary level
as explicit missing/degraded evidence.

## Measured guard behavior

The successful full run instrumented 98 confident/event-selection fit calls. Their wall-time
distribution was min 0.758 s, median 5.0008 s, p90 5.0017 s, p95 5.0027 s, and max 5.0089 s; 87
calls timed out. It also instrumented 225 weak-group calls, with matching 225 starts and ends. Their
distribution was min 0.346 s, median 5.0001 s, p90 5.0006 s, p95 5.0015 s, and max 5.0591 s; 169
timed out. The small overrun is cooperative-check and SciPy-return overhead, not another unbounded
integration.

The guard uses a monotonic deadline scoped by a context variable, checked before and inside RK4
integration. Nested helpers share the outer deadline. Candidate and weak fits return a
`blocked:segment_budget_exceeded` `FlightSegmentFit`; weak timeouts are appended rather than
discarded by plausibility filtering. The solved artifact, manifest, and chain result propagate the
typed counts and missing reason.

## Alternatives evaluated

1. **Bounded wall-clock guard (implemented):** the only option here that establishes a hard safety
   invariant without claiming a numerically different answer is more accurate. It preserves normal
   serialization when it does not trip and fails closed with typed missing evidence when it does.
2. **Adaptive/coarser or vectorized integration (not implemented):** batching predictions or using
   adaptive integration could remove much of the Python/RK4 cost. It changes numerical behavior and
   needs its own accuracy, physical-sanity, and byte/parity evaluation. It is an optimization follow-up,
   not a replacement for a safety bound.
3. **Candidate pre-filtering (not implemented):** a temporal/spatial gate or total-pool cap would
   reduce association work, but can remove the correct hypothesis and still cannot guarantee a bound
   for adversarial spans. It also needs frozen accuracy scoring before use.

## No-trip parity

The Wolverine internal-validation clip was run with an effectively unbounded budget and again with
a non-tripping 30 s budget. Both runs made 43 fit calls with zero timeouts. Run wall times were
80.756 s and 78.662 s; segment median/max times were 0.7236/14.3779 s and 0.6739/13.2754 s.
`cmp` returned 0 for all five substantive artifacts:
`ball_track_arc_solved.json`, `ball_arc_render.json`, `ball_flight_sanity.json`,
`events_selected.json`, and `ball_bounce_candidates.json`. Their SHA-256 values are recorded in
`wolverine_byte_identity.json`. `ball_chain_manifest.json` differs only in the five absolute output
directory path strings; all referenced artifact hashes are identical.

## Verification and honest limits

Focused solver, chain, synthetic pathology, weak pathology, and real-artifact tests: **77 passed in
49.01 s**. The mandatory wide suite completed with **3,719 passed, 8 failed, and 24 skipped in
2,796.84 s**. All eight failures are managed-sandbox socket-bind denials: six TCP loopback review
server tests and two AF_UNIX persistent-worker tests. The exact eight node IDs reproduce with the
same `PermissionError: [Errno 1] Operation not permitted` in a git-archive snapshot of HEAD
`58b10fdb651a6c40481c6b0de51e263ba1f31ad5`; proof is in `head_socket_failures.stdout`.

This change does not recover the 187 missing full-game segments, establish 3D accuracy, consume
ground truth, or alter any selected model/default stack. The production 5 s default can abstain on
otherwise valid clips whose individual fits exceed 5 s; Wolverine parity used a deliberate 30 s
non-tripping budget to prove only that the guard is inert when it does not trip. The cooperative
deadline had a measured worst overrun of 0.0591 s, but it is not an OS-level preemptive kill for a
future blocking native call. The repository storage audit remains red for 76 absent allowlisted
historical files and reports zero unknown large tracked or untracked files. `VERIFIED=0` remains
binding. There is no best-stack configuration knob and the BEST-STACK DELTA is **(c) none**.
