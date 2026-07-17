# BALL arc audio-onset soft anchors: final lane report

## Ruling

`objective_result=PARTIAL`. Arc fit coverage recovered materially in every pre-registered preset, but
**all three presets are KILLED** because the unchanged flight-sanity gate reported violations. Fit is not
accuracy, no GT exists, and every output remains preview-band with `VERIFIED=0`.

The taxonomy verdict is **needs-typed-anchors**, not fixable by further audio preset selection. Review-only,
`not_gate_verified` audio onsets can safely bound search pools, but cannot say that a contact occurred,
reset flight topology, constrain z to the ball radius, or become a flight-sanity anchor. No new presets were
added and no thresholds were tuned after scoring.

## Frozen rules and coverage definitions

`PRESET_REGISTRATION.json` was locked before coverage scoring. All presets use only `corrected_time_s`
inside rally-active spans derived from visible primary ball-track observations. Score-priority NMS enforces
the registered minimum spacing. Selecting all 2,309 raw onsets (about 3.3/s) would create degenerate
sub-second chunks and was prohibited.

In-rally frame coverage is the number of input frames inside a pre-registered rally-active span and any
emitted `status=fit*` segment, divided by all input frames in those spans. Total-segment fit fraction is the
number of emitted solver segments with `status` beginning `fit`, divided by all emitted solver segments,
including weak segments.

## Full per-preset results

| preset | locked rule | selected | fitted / total | segment fit | in-rally frames | frame coverage | wall | violations | provenance | kill |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| conservative | score >= 0.10, spacing >= 2.0 s | 168 | 53 / 371 | 14.29% | 3,362 / 17,914 | 18.77% | 39.69 min | 16 | 181 soft segments, 0 missing/invalid | **KILLED** |
| balanced | score >= 0.06, spacing >= 1.75 s | 202 | 85 / 361 | 23.55% | 5,312 / 17,914 | 29.65% | 38.34 min | 18 | 214 soft segments, 0 missing/invalid | **KILLED** |
| broad | score >= 0.03, spacing >= 1.5 s | 241 | 123 / 367 | 33.51% | 7,826 / 17,914 | 43.69% | 37.19 min | 18 | 253 soft segments, 0 missing/invalid | **KILLED** |

The frozen baseline was 1/188 fitted segments (0.53%). Every full run stayed below the 45-minute CPU
target with the unchanged 5 s production guard. Over-budget child segments remained typed
`segment_budget_exceeded` outcomes.

Baseline segment duration was min/median/p90/p95/max 1.63/33.83/81.13/89.07/111.93 s. For segments
actually created by soft splits, those distributions became:

| preset | min | median | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|
| conservative | 0.08 s | 3.15 s | 6.44 s | 7.31 s | 16.71 s |
| balanced | 0.13 s | 2.69 s | 5.01 s | 6.09 s | 15.53 s |
| broad | 0.13 s | 2.31 s | 4.45 s | 5.18 s | 12.84 s |

Thus the long hard-anchor pool is partitioned to rally-scale where soft evidence is available. The overall
emitted-segment max still remains 111.93 s because unsplit/weak fallback spans remain in the artifact; this
lane did not hide or relabel them.

## Mandatory violation taxonomy

`violation_taxonomy.json` contains all 52 failed gate segments with preset, flight-sanity segment id,
frame/time span, reasons, interior soft-boundary IDs, distance from failed motion checks to the nearest
boundary, violating solver segment IDs/statuses, weak flags, outside-frame counts, and maximum court
overage.

| preset | split landed mid-flight | bridged unmarked direction change | weak fit passed through | anchor-semantics structural | total |
|---|---:|---:|---:|---:|---:|
| conservative | 1 | 0 | 0 | 15 | 16 |
| balanced | 3 | 1 | 0 | 14 | 18 |
| broad | 5 | 0 | 0 | 13 | 18 |
| **all** | **9** | **1** | **0** | **42** | **52** |

Classification is evidence-based and deterministic: a weak flag takes precedence; a failed direction/speed
check within five frames of an interior soft boundary is `split-landed-mid-flight`; a motion/topology failure
without a nearby boundary is `bridged-unmarked-direction-change`; remaining non-weak free-depth/BVP
fallback geometry is `anchor-semantics-structural`. Every failing frame was non-weak; no failure was softened
into a weak-fit explanation.

The conservative-to-balanced trend is causal evidence against more audio threshold shopping. Balanced adds
34 untyped boundaries, 32 fitted segments, and 10.89 percentage points of frame coverage. Those extra fits
also make hard flight-sanity spans 0 and 15 evaluable; both become outside-court structural failures, raising
violations from 16 to 18. Denser broad splitting raises coverage again but holds at 18 failures and adds more
motion discontinuities adjacent to splits. Typed Track G contact/event anchors are needed to mark legal
flight resets; audio rules alone cannot supply that semantic fact.

## Default-off and no-audio boundary

With no soft anchors supplied, five substantive artifacts were byte-identical to adopted guard commit
`af6b8d40f` on Wolverine and five were byte-identical on a rebased real-demo slice (source frames
4750-4899). Both authoritative parity checks used the adopted guard lane's non-tripping 30 s comparison
method. Wolverine has no audio, so `soft_split_boundaries=()` and behavior is unchanged.

The boundary is plain: this recovery path currently applies only to audio-bearing captures. All product
captures have audio; internal Wolverine cards do not. It cannot improve the no-audio cards.

A separate first-2,100-frame diagnostic at the production 5 s cutoff was only 3/5 byte-identical because two
fits landed on different sides of the cooperative time limit under load. This known cutoff sensitivity is
recorded in `no_soft_byte_identity.json`; it is not used as parity evidence and the production 5 s default was
not changed.

## API handoff and BEST-STACK DELTA

Track C's additive API is `run_default_ball_arc_chain(..., soft_split_boundaries=())`, accepting typed
`SoftSegmentBoundary` objects. Omitted or empty input takes the exact old path and emits no new keys.
Non-empty input records typed provenance on each affected segment and in the chain manifest. Details and
forbidden semantics are in `API_HANDOFF.md`.

BEST-STACK DELTA is **(b) PENDING default-off `soft_split_boundaries` entry**. No best-stack, config, runner,
or stage file was edited. Because every preset is killed, Track C should not activate the input. A future
default-off surface is appropriate only after typed event anchors produce a zero-violation frozen-gate run.

## Verification and honest limits

- Focused solver/chain/guard/sanity suite: **85 passed in 24.13 s**, zero stderr.
- Wide racketsport suite: **3,758 passed, 37 failed, 24 skipped in 45:36**. Eight failures are the known
  managed-sandbox socket denials. The other 29 are outside this lane and cluster in dirty best-stack,
  court-eval/fixture, scaffold, and storage-policy surfaces, but were not proven to fail at HEAD in this turn.
  The wide suite is therefore not green and `failures_all_preexisting=false`.
- Audio onsets remain review-only, `not_gate_verified`, and untrusted for contact. Higher fit coverage only
  demonstrates bounded solver recovery, not accuracy.
- pb.vision demo data was used only as an R&D reference, never GT, training data, or redistributed output.
- No promotion, branch, commit, runner edit, guard bypass, or physics-gate modification occurred.
