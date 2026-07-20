# farline_diag_20260720 result

`RECOVERED` on this sixth real-world clip. `VERIFIED=0`; clip-specific preview
evidence only; T14 remains default-off and unwired; the frozen five-venue
verdict is unchanged.

## Requested record

```json
{
  "far_centerline_baseline_accept_count": 4,
  "pooled_recovered": true,
  "support_frames": 63,
  "residual_px": 0.35691255314973347,
  "would_pass_gate": true,
  "reason_if_not": null,
  "evidence_jpegs": [
    "runs/lanes/farline_diag_20260720/evidence_frames/far_centerline_frame_000235_annotated.jpg",
    "runs/lanes/farline_diag_20260720/evidence_frames/far_centerline_frame_005055_annotated.jpg",
    "runs/lanes/farline_diag_20260720/evidence_frames/far_centerline_frame_011167_annotated.jpg"
  ]
}
```

`residual_px` is T14's optimize-sample geometry-fit p90, not independent GT
error. The recovered segment's residual to the frozen seed projection is
1.484 px mean / 1.702 px p95, with 0.9177 visible fraction.

## Baseline: 96 evenly spaced frames

| Semantic line | Accepted frames |
|---|---:|
| far_baseline | 28 |
| far_centerline | 4 |
| far_nvz | 79 |
| left_sideline | 96 |
| near_baseline | 95 |
| near_centerline | 51 |
| near_nvz | 96 |
| net floor line | 14 |
| right_sideline | 96 |
| top_net | 22 |

The current local legacy-Hough aggregate is already
`auto_calibration_ready=true` at 96 frames. This means the requested premise
that the frozen baseline would be `0/96` did not reproduce locally.

The pulled production artifact itself is unambiguous: `0/7` far-centerline
support and `auto_calibration_ready=false`. A current local rerun of the same
seven indexes returns `1/7` (frame 5584) and ready=true. OpenCV 5.0 and a second
OpenCV 4.11 environment produced the same local counts, so this is not random
Hough behavior inside this run. The VM's exact decoded pixels/runtime were not
preserved, so the cause of the pulled-vs-local discrepancy is unproven; it is
recorded as a reproducibility caveat, not silently treated as a reproduction.

## T14 pool

The unchanged-default `seed_guided_paired_edges` provider recovered the actual
painted far centerline:

- 51 optimize frames + 12 independently held-out frames = 63 support frames;
- all 41 far-centerline sample locations passed in both optimize and held-out
  pools;
- temporal MAD 0.0289 px; geometry-fit p90 0.3569 px;
- median paint-band contrast about 100.44 gray levels, median band width 3.15
  px, and median edge strength 50.53;
- frozen semantic selector confidence 0.9675; seed residual 1.484/1.702 px
  mean/p95.

The visual overlays show the green pooled segment centered on the real white
paint. Frames 5055 and 11167 are especially diagnostic: the local legacy Hough
path accepts no far-centerline segment, while the paired-edge samples cover the
paint band.

## Gate counterfactual

Adding only the recovered far-centerline observation to the immutable pulled
production evidence changes the exact gate aggregate to:

- `auto_calibration_ready=true`;
- missing required lines `[]`, missing net IDs `[]`, reasons `[]`;
- mean residual 3.6499 px, p95 10.2531 px.

Therefore it would clear the evidence-readiness condition that caused
`_court_calibration_needs_correction` to block tracking. No runner code was
changed, so the current product still does not perform this recovery.

The T14 pool as a whole accepted seven floor lines but independently abstained
on its own pooled `near_centerline` because geometry p90 was 4.7404 px over the
1.75 px bar. That does not invalidate this additive recovery: the pulled frozen
evidence already contains an accepted near-centerline, and only the missing far
line is added for the gate counterfactual.

## Verification

- Video SHA-256 matches
  `5085ae6ed0813b2b05ce1d6fe752423506cdc3fb78ca751d185403889b47b181`.
- Reversed-frame T14 rebuild is canonical-byte-identical:
  `c611870ef6240209c76700937e8449ae1d19822e61d90dd0892d87720b8c4713`.
- Focused tests: `45 passed in 6.50s` across court line hardening, automatic
  court evidence, and the evidence gate.
- Shared code was read-only; no commit, branch, push, GPU, or runner wiring.
