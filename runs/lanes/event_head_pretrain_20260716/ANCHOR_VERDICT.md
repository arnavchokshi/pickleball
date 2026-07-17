# ANCHOR VERDICT — DO NOT INGEST (Track G2 manager, 2026-07-17)

**Bottom line for Track A: do NOT ingest `anchors/pbvision_11min_event_head_anchors*.json` as
anchor evidence. The artifacts are committed as EVIDENCE OF A DOMAIN GAP, not as a deliverable.**
The defining deliverable of this lane is therefore the checkpoint + `SCALE_UP_SPEC.md`, not the
anchors. Reported straight, per the owner's honesty rules.

## What the head does on public tennis (GOOD, measured)

Matched 64-frame windows, 16 clips, 42 GT events, threshold 0.5, tolerance ±2f:
HIT tp6/fp0 (recall ~22%), BOUNCE tp3/fp0 (recall ~20%) — **9 predictions, ZERO false positives.**
High-precision / low-recall. (See `eval/matched_window64_eval.json`.)

## What the head does on the pickleball demo video (DISQUALIFYING, measured)

| metric | value | plausible? |
|---|---|---|
| HIT candidates @thr 0.5 | **4,990** over 697.4s = **7.16/s** | no — a real game has ~200-400 contacts total |
| BOUNCE candidates | 110 | — |
| median inter-HIT gap | **4 frames** (stride was 2) | no — the head fires at ~every other window position |
| seconds containing ≥1 HIT | **680 of 697 = 98%** | no — the video has 41 rallies with dead time between them |
| max HITs in one second | 10 | no |
| score range | min 0.5001, median 0.6725, max 0.9542 | confidently wrong |

Merging with a wider NMS radius does not rescue it — the activation is a near-uniform carpet, not
tight clusters around discrete events:

| merge radius | distinct HIT clusters | rate |
|---|---|---|
| 2f (as emitted) | 4,990 | 7.16/s |
| 5f (0.17s) | 327 | 0.47/s |
| 10f (0.33s) | 103 | 0.15/s |
| 15f (0.50s) | 49 | 0.07/s |

The `nms_radius_frames=2` config IS too tight relative to `stride=2` (a real fix for a future lane:
contact spacing warrants ~10-15f), and it inflates the raw count. **But that is not the root cause:**
98% temporal coverage means the head cannot separate contact from non-contact on pickleball pixels
at all. Tennis-broadcast features did not transfer.

## The audio cross-check cannot rescue them (and is itself informative)

Per the owner's multi-signal directive, every candidate carries an evidence vector
(`anchors/pbvision_11min_event_head_anchors_enriched.json`, built by `logs/enrich_anchors_evidence.py`):

- 3,580 of 5,100 candidates (**70.2%**) have an audio onset within ±0.15s.
- **Chance baseline: ~99.3%.** There are 2,309 review-only onsets across 697.4s (mean spacing
  0.302s), so a 0.30s-wide window around *any arbitrary timestamp* almost always contains one.
- Observed co-location is therefore **at/below chance** — the audio channel adds **zero
  discriminative information** for these candidates.

**This independently corroborates the owner's own observation** (neighboring-court audio bleed):
audio onsets on this video are so dense (3.3/s) that they are near-uniform, which is precisely why
audio-only anchors "keep tripping physics" on Track A's side. It is direct evidence FOR the
multi-signal design and AGAINST any single-signal anchor class.

Ball-track kink was computable for 3,205 of 5,100 (62.8%); `wrist_swing_proximity` is unavailable —
no BODY/pose artifacts exist for this video.

## Honest conclusion

1. The checkpoint is real and behaves sensibly **in its training domain** (tennis, high-precision).
2. Its **zero-shot transfer to pickleball is a failure**, not a weak success. Do not dress it up.
3. The cause is the one `SCALE_UP_SPEC.md` quantifies: **2.4% label reach / 18.1% media coverage /
   one-window-per-row** — the head saw a rounding error of the corpus and zero pickleball.
4. The path is the scale-up (levers 1-3) followed by the pickleball fine-tune the scaffold already
   supports (`finetune_event_head.py`, gated on the owner's 300-clip review drop). Anchors should be
   re-attempted only from a fine-tuned checkpoint that first demonstrates a plausible firing rate on
   the demo video (a cheap pre-flight: candidates/second must be ~0.3-1.0, not 7).
5. VERIFIED=0. No promotion. Nothing here enters the default stack.
