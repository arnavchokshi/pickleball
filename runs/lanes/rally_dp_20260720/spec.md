# rally_dp_20260720 — pre-registered event-sequence DP contract

## Posture and file fence

- `VERIFIED=0`; research-only, default OFF, no best-stack or pipeline wiring.
- This lane performs sequence selection over **saved** event-head logits or probabilities. It does not run model inference, fit whole-rally geometry, synthesize timestamps, or mutate raw scores.
- Raw and selected predictions are always retained side by side. Every selected anchor points to its saved frame and class column through `score_trace`.
- Dense, exhaustively typed, source-disjoint public rallies are the only admissible judge. Sparse owner/protected rows and event-centered partial windows are invalid DP judges.
- Frozen implementation files: `threed/racketsport/event_head/sequence_dp.py`, `scripts/racketsport/apply_event_sequence_dp.py`, and `tests/racketsport/test_event_sequence_dp.py`. No existing source file is changed.

## Input contract

Schema version 1 uses:

```json
{
  "schema_version": 1,
  "artifact_type": "event_head_sequence_input",
  "verified": false,
  "class_names": ["background", "HIT", "BOUNCE"],
  "ground_truth_policy": "none | dense_exhaustive",
  "clips": [{
    "clip_id": "stable identifier",
    "fps": 29.97,
    "probabilities": [[0.9, 0.08, 0.02]],
    "rally_spans": [{"rally_id": "r1", "start_frame": 0, "end_frame": 1}],
    "hit_side_by_frame": [null],
    "ground_truth_complete": true,
    "ground_truth": [{"frame": 0, "class": "HIT", "side": "A"}]
  }]
}
```

- Exactly one of `probabilities` or `logits` is required per clip. Scores have class order `background,HIT,BOUNCE`; logits are converted deterministically with stable softmax.
- Rally spans are non-overlapping, half-open clip-local frame intervals.
- `hit_side_by_frame` is optional and accepts only `A`, `B`, or `null`. A/B mean team/court side, not individual identity. Missing side causes abstention from the alternation term.
- `ground_truth_policy=dense_exhaustive` requires `ground_truth_complete=true` for every clip and is the only mode that emits precision/recall comparisons.

## Frozen constraints and provenance

These values were registered before any dense-judge score and are not CLI-tunable:

| Parameter | Frozen value | Evidence/rationale |
|---|---:|---|
| Low-threshold raw candidate floor | `0.05` probability | Existing T20 step-16918 frozen eval threshold; candidate generation must trace to the saved low-threshold head output. |
| Raw peak NMS radius | `2` frames | Existing `event_head.matcher.peak_pick` contract, preserved for raw comparability. |
| Minimum inter-HIT spacing | `0.5 s` | MonoTrack reports no two TrackNetV2 hits within 0.5 s and uses 0.5 s as its empirical spacing constraint. This is an external starting hypothesis, not a claimed pickleball constant. |
| Selected and prerequisite raw event-rate band | `0.3–1.0 events/s` | Owner/advisory physical band. The upper edge also matches MonoTrack's published approximation that a D-second rally has about D hits. |
| Maximum HIT count | `floor(1.0 × rally_duration_s)` | Source-faithful D-second/D-hit upper prior. |
| Same-side alternation | bounded `1.0` log-odds penalty | MonoTrack uses hard opposing-player alternation. Advisory D2 requires team/court-side alternation to be soft for doubles; this pre-registered one-unit penalty can never veto a saved candidate. Missing side abstains. |
| Same-frame typed anchors | at most one | HIT and BOUNCE are mutually exclusive event-head classes at one timestamp. |

Primary analogous source: [MonoTrack paper, arXiv:2204.01899](https://arxiv.org/abs/2204.01899), especially §4.4 and Table 1. It reports HitNet recall `78.1% → 94.3%` after constrained optimization, but that badminton result is not treated as pickleball verification.

The objective is the sum of per-candidate `log(p_event / p_background)` minus the bounded same-side penalties. Dynamic programming selects the global best feasible subsequence. Ties resolve lexicographically by saved candidate order, so output is deterministic.

## Fail-closed behavior

- CLI default, explicit `--off`, and in-memory `enabled=False` are identity passthroughs. CLI OFF copies the exact input bytes.
- Enabled mode refuses to emit typed anchors for a rally when its **raw** low-threshold candidate rate is outside `0.3–1.0 events/s`; DP may not launder a degenerate head.
- It also refuses when discrete count bounds or saved candidates make the constraints infeasible. It never creates a timestamp to satisfy a lower rate.
- An ineligible rally has `selected_predictions=null`, no downstream typed anchors, and a machine-readable reason.

## Pre-registered acceptance

1. Determinism: two enabled runs over identical bytes produce byte-identical JSON.
2. OFF identity: default and `--off` outputs equal the input byte-for-byte.
3. Synthetic proof: known true events are all retained; constraint-violating false firings are removed; hard-constraint audit returns zero violations.
4. Dense public judge only: report raw and DP typed precision, recall, F1, and timing at frozen 1/2/5-frame tolerances only when full rally scores, spans, and exhaustive GT are present.
5. Survival remains stricter than implementation: dense-rally typed F1/recall must improve with precision and timing non-regression. Product activation additionally requires a small untouched exhaustive pickleball-rally holdout.

## Frozen local evidence status

The pulled step-16918 JSON is not DP-scoreable: it contains aggregate metrics and per-clip counts/maxima, but no per-frame logits/probabilities, timestamps/types, or rally spans. Its 510 predictions cover 50 × 64-frame windows at 29.97002997 fps (`106.7733 s`), an aggregate `4.7765 events/s`, already outside the pre-registered raw gate. Those windows also come from the event-centered evaluation policy rather than full dense rally rows. No DP precision/recall delta is claimed from that artifact.

