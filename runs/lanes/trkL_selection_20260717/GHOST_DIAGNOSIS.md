# Ghost diagnosis — the 4 surviving wolverine "spectator" FPs under RF-DETR-L variant P

Track L manager, 2026-07-17. CPU artifact forensics on the pulled VM artifacts
(`runs/lanes/trk_rfdetr_prod_20260716/vm_rerun/pulled/rfdetrflip_pull.tar.gz`, sha-manifested).
Scoring semantics taken from the frozen scorer's OWN internals (imported, not reimplemented);
the as-pulled wolverine variant-P row reproduces the card exactly on this machine
(0.8036 IDF1 / 0.7233 cov4 / 1 sw / 4 spectFP / 0.1244 near-miss rate) — fixed-input *scoring*
is platform-stable; it was association *production* that diverged on Mac (standing finding
unchanged: card rows are made on GPU-class VMs only). VERIFIED=0. Nothing here is a card row.

## Verdict in one line

**The 4 "spectator" FPs are not spectators and not detector output — they are 4 frames of a
42-frame synthetic interpolation bridge that the global association manufactured when it
stitched two different players' tracklets into one track.** The detector did its job; the
margin gate did its job; the cleaning layer manufactured the ghost itself.

## The event, frame by frame (wolverine, variant P, track 4)

- f0-44: track 4 = GT player 1 (43 matched frames, conf 0.87-0.93, world ~(-1.5, -3.2)).
- f44: last real detection (conf 0.82). The real player continues to exist; the pool loses him
  only briefly — but a NEW track (id 1) is born at f55 and takes GT player 1 from f55 on.
  That handoff is the card's single ID switch (scored at f58, 16-frame gap, IoU 0.596).
- f45-86: **42 consecutive synthetic frames, conf exactly 0.3500 each** (the interpolation
  cap), bbox sliding linearly ~3.6 px/frame while the projected footpoint marches
  **0.23 m/frame (~7.4 m/s sustained) in a dead-straight line across the entire court,
  through the net, from (-1.48, -3.26) to (1.62, 6.68) — 10.4 m.** No pickleball player does
  this; no detector saw it; it is `_interpolate_detection` output.
- f87-299: track 4 = GT player 4 (far baseline, real detections again, conf 0.77-0.83).
- Scoring of the bridge: 4 frames (59-62) overlapped NO GT player at IoU 0 on full-GT frames →
  `true_spectator_or_background_false_positives = 4`. Most of the other bridge frames grazed
  real players → near-miss FPs (the bridge is why wolverine's near-miss rate 0.1244 breaches
  the 0.10 gate; strip it and the rate is 0.0986). Footpoints of the ghost frames are ON the
  court (excess_m = 0.0, sweeping the kitchen) — **structurally invisible to any footpoint
  gate, hard or soft, at any margin.**

## Why every existing defense missed

1. **Court-margin 1.0 m gate** (`drop_outside_court: true` in the production authority config)
   runs on POOL detections, pre-association. It correctly removed the real high-confidence
   spectators (detbench zero-shot saw 16 of them). The bridge is created AFTER the gate and its
   interpolated footpoints are on-court by construction.
2. **Conf floor**: synthetic frames carry conf = min(endpoints, 0.35) = 0.35, above any
   deployed floor; and the floor applies at the pool, not post-association. The preregistered
   conf-0.30 single-shot made wolverine WORSE (16 spectFP, 2 sw) for exactly this reason:
   higher floor → more real far-player dropouts → more gaps → more/longer bridges. The
   conf030 report's mechanism guess ("spectator detections are high-confidence") is hereby
   CORRECTED by forensics: its 16 surviving FPs are also bridge frames — the conf030 wolverine
   tracks contain the IDENTICAL bridge (f44-86) plus the same four later synthetic runs on its
   track 3 (runs (137,166),(168,174),(185,188),(268,292) — byte-level signature match with
   variant P's track 4).
3. **Speed guard scales with gap length**: `allowed = merge_distance_slack_m (1.25) +
   max_gap_fill_speed_m_s (7.0) × dt`. At the production `max_gap_fill_frames: 48`
   (authority config; module default is 24), a 43-frame gap buys dt = 1.43 s → 11.28 m
   allowance — more than the 10.4 m the bridge traveled. A per-gap DISTANCE cap does not exist.
4. **Appearance evidence could not veto**: in `_backfill_player_cost`, a missing embedding
   degrades to a NEUTRAL appearance cost of 0.5, and appearance is a weighted term in a cost
   soup (`appearance_weight 1.0 · cost + 0.35 · motion + …` vs `max_merge_cost 2.0`), never a
   hard refusal. Synthetic frames legitimately carry no embedding (global association even
   documents this), so the stitch decision leaned on motion/side terms.
5. **Provenance is stripped at export**: `player_id_repair._interpolate_detection` stamps
   `conf_source="interpolated_endpoint_min_capped_0_35"`, but exported `tracks.json` frames
   carry only bbox/conf/t/world_xy. Downstream consumers (and the scorer) cannot tell
   synthetic from real. Synthetic frames even COUNT TOWARD cov4 (scorer counts prediction
   cardinality: matched + unmatched-unignored), so the bridge *pads* cov4 by ~32 frames —
   the current 0.7233 contains fake coverage.

## Embedding forensics (OSNet x1_0, pinned production checkpoint, production crop padding 8px)

Mean cosine distances between real-frame crops (CPU probe, `diagnosis/osnet_stitch_probe.json`):

|                        | T4 pre-stitch (GT1) | T4 post-stitch (GT4) | T1 (GT1 continuation) |
|---|---|---|---|
| T4 pre-stitch (GT1)    | **0.13** (within)   | **0.448**            | 0.304                 |
| T4 post-stitch (GT4)   | 0.448               | **0.10** (within)    | 0.424                 |
| controls (T2, T3 self) | 0.11 / 0.16 within  | cross-identity range 0.32-0.55 | |

The stitch pair (0.448) sits squarely in the cross-identity band; the legitimate re-bind pair
(GT1 early vs GT1 later, 0.304) sits clearly below it. OSNet evidence, consulted as an
open-set veto instead of a weighted cost, refuses this stitch while accepting the true
re-bind — on this clip. Honest caveats: N=1 clip, 5-6 crops/segment, and the margin
(0.30 vs 0.45) is real but not enormous → thresholds must be pre-registered with a defer band
and fused with kinematic/court evidence (owner's rule: no single signal decides).

## Counterfactual decision table (frozen scorer, fixed VM-produced tracks, CPU diagnostic — not card rows)

| wolverine arm | IDF1 | cov4 | sw | spectFP | near-miss rate | HOTA |
|---|---|---|---|---|---|---|
| YOLO26m baseline (card) | 0.8516 | 0.7600 | 0 | 0 | — | 0.8611 |
| RF-DETR-L variant P (card) | 0.8036 | 0.7233 | 1 | 4 | 0.1244 ✗ | 0.8021 |
| CF1: strip the synthetic bridge only | 0.8141 | 0.6167 | 1 | **0** | 0.0986 ✓ | 0.8112 |
| CF3: + slot re-bind (GT1's continuation returns to slot 4; post-stitch segment split off) | **0.8519** | 0.6167 | **0** | **0** | 0.0986 ✓ | **0.8614** |

CF3 = the behavior layer B is designed to produce, simulated here with GT knowledge as an
upper bound. Every owner-hated axis zeroes and IDF1/HOTA land at/above the YOLO baseline.
The single remaining deficit is cov4 0.6167 — which is the honest number once fake padding is
removed; rebuilding it with REAL detections is layer A+B fusion's job (identity-conditioned
pool recovery; the raw pool is exported at min_conf 0.0 — recall is there to be used).
Burlington is structurally untouched by all of this: 6 synthetic frames total, longest run 3
frames / 1.14 m — no veto rule fires; its all-clean row is preserved by construction.

## Code seams (exact)

- `threed/racketsport/player_global_association.py:1147` (`_fill_short_gaps` → synthesis),
  `:1161-1173` (`_interpolate_detection`, conf cap 0.35), `:1069-1097` (cost soup + neutral-0.5
  missing-embedding fallback), config `:35-63`.
- `threed/racketsport/player_id_repair.py:522,536-551` (same interpolator, provenance stamp
  that dies at export).
- Production operating point that let it through: `raw_pool_authority_summary.json` config —
  `max_gap_fill_frames: 48`, `max_gap_fill_speed_m_s: 7.0`, `merge_distance_slack_m: 1.25`,
  `max_merge_gap_frames: 240`, `min_conf: 0.0`, margin gate at pool stage only.

## Reproduction

- `diagnosis/diagnose_ghosts.py <clip> <tracks.json> <gt.json> <out.json>` — enumerates every
  unmatched prediction with the frozen scorer's own matching; outputs are committed
  (`diagnosis/{wolverine,burlington}_rfdetr_l_p_diag.json`).
- `diagnosis/counterfactuals.py` — CF1/CF3 rows above.
- Extracted artifact trees are NOT committed (reproduce: untar the two pulled tarballs in the
  trk_rfdetr_prod lane; sha256 manifests live beside them).
