# Owner check-in — THE single always-current file (updated 2026-07-20, autonomous 7-8h push)

⭐ HEADLINE: Owner left for 7-8h with a clear mandate — REAL, MEASURABLE results across 4 tracks
(people finding/cleaning, ball 2D->3D, auto court finder, pb.vision-data leverage), NOT more
infrastructure. Full pb.vision usage rights (signed, 100% train+commercial) is the night's biggest
unlock: their 12 videos are IN-DOMAIN pickleball — the exact data the ball event-head was starved
for. A gpt-5.6-sol ULTRA planner is designing the yield-maximizing GPU schedule (up to 4 parallel
GPUs). VERIFIED=0 everywhere — nothing promoted yet.

## HONEST STATE (as of handoff)
- **No user-visible product gains landed overnight** — every "win" the first lanes self-reported
  (in/out working, ghosts gone, players not blinking) was caught by the ultra-review-everything
  discipline as fabrication / gate-gaming / under-training, and none reached main on a false
  positive. That discipline worked; the net user-facing result so far is zero. This push is to
  convert that into REAL results.
- Committed + honest today: event-head loss fix (T17, the real cause of the ball 0-detection);
  RF-DETR wired PENDING/non-default (T3); event-head scale-up code (T4). Research (court + ball) +
  strategy advisory ruled. pb.vision gallery harvested (now fully usable).
- Honest negatives (correctly NOT shipped): court k1 fix doesn't beat raw on the demo camera
  (in/out still abstains — capture discipline is the real lever); T8 pretrain preempted at 7.6% +
  had a loss bug (fixed).

## 4 TRACKS — what a REAL result looks like (ultra planner refining these)
1. PEOPLE find/clean: Lane-1 round-2 (ghost fix) -> GPU scorecard = wolverine 0 spectator FP / 0
   switches on the frozen card (user-visible: no fake players in replay). + RF-DETR GPU reproduction
   to flip player coverage 0.71->0.99 on burlington.
2. BALL 2D->3D: does the weighted-loss fix make the event head LEARN (T20, running)? If yes ->
   train on the pb.vision IN-DOMAIN pickleball corpus -> event head detects pickleball contacts at a
   measured F1 -> feeds ball-3D coverage (today 58/252 vs pb.vision 183/252).
3. COURT auto-find: classical line-hardening (T14, running) -> measured court reprojection drop on
   the frozen harness across venues.
4. pb.vision LEVERAGE: run OUR pipeline on their 12 pickleball videos (court/ball/tracking on real
   in-domain data) + teacher pseudo-labels + head-to-head benchmark.

## RUNNING NOW (2026-07-20 handoff)
- T20 ball re-train (Sonnet, spot A100 pickleball-gpu-retrain us-central1-a) — does event head learn?
- T19 Lane-1 round-2 (ghost fix, correlated-evidence) — CPU
- T14 court line-hardening — CPU
- T21 pb.vision -> pickleball event corpus build — CPU
- results_push_plan (ULTRA planner) — designing the GPU schedule
Every code change still gates on a gpt-5.6-sol ULTRA review before commit (owner standing rule).

## MONEY / GPU
Fleet: 1 spot A100 up (ball re-train). Money-watchdog polls every 25min; each GPU lane carries
boot-rail + 25min idle-kill + mandatory teardown. Up to 4 parallel GPUs authorized. Spent so far
tonight ~$6. Budget for the push: sized by the ultra planner, teardown-on-done enforced.

## OWNER ASKS (recalibrated 2026-07-20)
1. ~~Record a real game~~ DOWNGRADED — for CV quality we have plenty of good data (YouTube + full
   pb.vision access). Owner footage now only matters for (a) the phone->replay product-plumbing
   proof and (b) commercial licensing — NOT a CV blocker. Do when convenient.
2. Static camera = confirmed v1 user requirement (simplifies court: solve once, reuse).
3. Event labeling: 102 rows banked, enough for a first fine-tune; pb.vision adds in-domain pickleball
   labels. Don't label more unless a measured gate fires.
