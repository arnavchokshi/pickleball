# Track C refinedstage handoff: default-off soft split API

The additive public chain entry point is:

```python
run_default_ball_arc_chain(
    ...,
    soft_split_boundaries: Sequence[SoftSegmentBoundary] = (),
)
```

`SoftSegmentBoundary` is defined in `threed.racketsport.ball_arc_solver` with required fields
`boundary_id`, `corrected_time_s`, `frame`, `onset_ids`, and `selection_rule_id`; its fixed
`anchor_class` is `audio_onset_soft`. `source_artifact` is optional but should be supplied by Track C.

Omitted or empty input takes the exact pre-existing code path and emits no new manifest keys. A non-empty
input is passed to the solver and recorded in both the solved artifact and chain manifest. Each affected
segment carries `soft_split_provenance`. The typed payload explicitly records `event_type=null`,
`world_constraint=null`, `counts_as_bounce_evidence=false`, and
`counts_as_flight_sanity_anchor=false`.

Track C owns runner parsing and construction of these objects. It must not map audio onsets into bounce or
contact candidates, z/radius constraints, or flight-gate anchors. The production guard remains the existing
5 s default. Any over-budget child segment remains a loud typed `segment_budget_exceeded` outcome.

Do not enable this input in refinedstage yet: all three pre-registered audio presets were killed by the
unchanged flight-sanity gate. The BEST-STACK DELTA is therefore **(b) PENDING**: a future
`soft_split_boundaries` chain input may be represented as a default-off entry after Track G provides typed
event/contact semantics and a zero-violation frozen-gate run passes. No config or runner file was edited in
this lane.
