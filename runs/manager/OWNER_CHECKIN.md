# Owner check-in — THE single always-current file (updated 2026-07-19)

⭐ HEADLINE: FULL-PARALLEL WAVE LIVE. 4 Codex lanes dispatched simultaneously (P0-I fabrication
fix + selection layers; static-CAL single-lock + k1 fix that should un-abstain in/out; RF-DETR-L
production integration; event-head scale-up code). Queue-#1 forensics DONE + RULED by manager:
the ball program does NOT reorder — only 5/24 fusion contact refusals overlap fabricated player
frames; the trained-event wall stays the blocker (=> your event labels are the highest-value
thing you can do today). Two bonus catches: true 4-player coverage is 0.520 not 0.7233
(synthetic padding 0.203), and exported world positions are piecewise-FROZEN (p2: 31 distinct
positions in 300 frames) — both routed into the live lanes. GPU legs (event-head train ~$2-4.5,
selection card eval ~$0.5, RF-DETR repro ~$1-3) fire as each code lane lands. VERIFIED=0 unchanged.

## Do these (easiest first — CVAT is ALREADY RUNNING, no setup needed)
0. **Keep this Mac plugged in, lid OPEN, all night.** The ONLY hard requirement — caffeinate is
   armed but lid-close sleep still kills lanes. The whole overnight board depends on it.
1. **STOP labeling events — 102 is enough.** Your 102 rows are banked; the first pickleball
   fine-tune runs on them tonight. Advisory ruling: only label more IF the fine-tune's measured
   G_val gate fires (I'll tell you). Do not open the events page again for now.
2. **Court tasks 88–91 are now OPTIONAL / deprioritized.** Advisory D1: the fastest court win is
   better classical line-detection + capture-discipline UX, NOT more corner labels — so 88–91 drop
   off the critical path. Do them only if you feel like it; they're now frozen eval fuel, not a
   calibration lever.
3. **2-min form (still worth it):** CoachAI Badminton Track-1 (free frame-precise event data):
   https://forms.gle/znfgo4Bvp3t9h8wk9
4. *(standing, #1 overall)* Record one real pickleball game on the product phone — still the single
   biggest unlock (we own ZERO pickleball footage); parked per your note you can't record now. It
   outranks any further old-pack labeling.

⚠️ **Honest correction from earlier tonight:** I over-sold the pb.vision-gallery pseudo-label idea
("multiply your 70 labels without touching CVAT"). The governance review (advisory D3) ruled it
OUT: their terms restrict this use, and training on their gallery would wreck our head-to-head
benchmark. The 12-video gallery is still a big win — as a 12× COMPARE-ONLY benchmark (~490 rallies)
and proof their system tracks no player pose (our wrist advantage is unique) — just not as training
data. Pseudo-labels, if ever, come from public/owned footage + our own models, never theirs.

## Best results so far (accuracy + time; ALL internal/protocol-caveated, none promotion-gated)
| Capability | Best accuracy | Best time | Note (evidence under runs/) |
|---|---|---|---|
| Ball 2D | F1@20 0.7248, recall 0.626, hFP 0.063 (zero-shot anchor, held-out Outdoor) | 25-35 fps A100 | label curve alive; gate F1>=0.90 |
| Ball 3D | 100% physics-plausible emitted; rally coverage 58/252 vs pb.vision 183/252 (Wolverine) | ~122s refined arc inside world stage | wall = trained contact/event detection (xref 20260719 re-confirmed: NOT fabrication-dominated) |
| Find people | raw detector >=4 players on 96.7%/78.1% frames (Burl/Wolv) | — | RF-DETR-L integration lane LIVE (burlington best-ever 0.9220/0.9933 pending GPU repro) |
| Track people 2D | mean IDF1 0.852; rev-12 margin flip worst 0.6425->0.8516 | 29.8-32.3 fps | TRUE wolverine cov4 = 0.520 (padding 0.203 was synthetic — P0-I); selection-layer fix lane LIVE |
| People 3D (BODY) | root-rel 59.7mm / PA 39.9mm; Track I foot-slide 4/4 <=30mm (6.7/5.6/6.3/6.8mm, scoped) | BODY 285.7s cold H100 | gate = independent world-MPJPE <=50mm (NS-02) |
| Court | ours 6.61px vs pb.vision 5.67px (frozen M4); line solve 2.61px; 15-pt reviewed 19.16px (zero-distortion defect) | calibration 1.33s/clip | k1-fix lane LIVE — expected: in/out stops abstaining on demo |
| Paddle | rectangle IoU 0.22-0.33, preview only | 0.39s/clip | no pose/contact GT; gold capture unlocks |
| Events | val F1@±2 0.3631 (T4 pretrain, starved); matched-window 9TP/0FP high-precision | — | scale-up = ~68x windows for $2-4.5; YOUR labels = the fine-tune |
| E2E speed | — | spine 523s cold / 487s warm (H100) | ~122s = intended refined-arc solve; NS-06 lever = reuse |

## Running right now (2026-07-19)
4 Codex lanes (all dispatched this session, logs under runs/lanes/*/log.txt):
trkL_selection_impl_20260719 (P0-I fix), static_cal_firstlock_20260717 (CAL lock + k1),
trk_rfdetr_integrate_20260717 (detector flip build), event_head_corpus_20260719 (scale-up code).
Manager rules on each report, then GPU legs: event-head train (A100-40/L4/T4 ladder, cap $10),
selection card eval (A100 micro ~$0.5), RF-DETR repro (~$1-3). Fleet EMPTY right now, $0/hr.

## Money
Today so far: $0 GPU. Planned this wave: ~$4-9 total across three short GPU legs, each with
boot-armed rail + teardown + list-confirm. gcloud auth confirmed active (project
gifted-electron-498923-h1).
