# Owner check-in — THE single always-current file (updated 2026-07-14, handoff for account switch)

⭐ HEADLINE: TRK "people disappearing" FIXED in the default stack (margin-1.0+OSNet WIRED_DEFAULT rev
12; worst-clip IDF1 0.64→0.85, preview-band). pb.vision fully decoded from your 11-min clip — their
ball-3D edge is TRAINED contact/event detection, nothing exotic; your "public data exists" hypothesis
was VINDICATED (~130k public hit/bounce events now on disk + 117 bootstrap pickleball labels). Ball
event-head training is the concrete path, gated on your 5-min spot-check. The 41-rally head-to-head
died on the Fable spend limit mid-run (no result — RE-RUN needed). Fleet EMPTY, orphan VM deleted.
FULL DETAIL for the next session: `runs/HANDOFF_20260714.md`. VERIFIED=0 unchanged.

## Do these (easiest first)
1. **Court labels, ~1h, highest value:** Docker Desktop -> CVAT up, then
   `.venv/bin/python cvat_upload/court_diversity_20260712/import_court_diversity_tasks.py`,
   label 4 shards x 25 frames per `cvat_upload/court_diversity_20260712/OWNER_GUIDE.md`.
   100 frames / 28 new venues — attacks the proven court diversity wall.
2. **Court tasks 88-91** in CVAT (already staged) — label whenever.
3. **Ball task 87** (350-frame uniform audit) — finish whenever; ingest is committed.
4. **Phone, 10 min:** (a) open DinkVision in landscape, tap the yellow record button — does
   recording start? (an audit session has waited 3 days on exactly this); (b) then say
   "stage fps test" and I stage a Wolverine replay URL — first real-device fps number ever.
5. **NS-01.2b physical proof (~15 min, signed device):** one 30s record -> upload -> open own
   replay on the real app.
6. **2-min data unlocks (from the event-data research — your hypothesis was right):** submit the
   CoachAI Badminton Challenge Track-1 access form (videos+hit-frames shipped together):
   https://forms.gle/znfgo4Bvp3t9h8wk9 — and optionally email the BFMD authors (arXiv 2603.25533)
   for their badminton hit+landing set.
7. **Gold capture half-day (standing):** paddle 6DoF, ball-3D depth, and BODY accuracy are all
   capped until this exists. `runs/lanes/ns021_goldcapture_20260709/OWNER_HALF_DAY_CHECKLIST.md`.

## Best results so far (accuracy + time; ALL internal/protocol-caveated, none promotion-gated)
| Capability | Best accuracy | Best time | Note (evidence under runs/) |
|---|---|---|---|
| Ball 2D | F1@20 0.7248, recall 0.626, hFP 0.063 (zero-shot anchor, held-out Outdoor) | 25-35 fps A100 (0.51-0.97x realtime, 4 internal clips) | label curve alive: 1k-label ckpt 0.6152 internal; gate F1>=0.90 (heldout_eval_ledger row 4) |
| Ball 3D | 100% physics-plausible emitted; rally coverage 58/252 vs pb.vision 183/252 (Wolverine) | inside world stage (~122s incl. refined arc solve) | wall = upstream 2D/event evidence, not the solver (research_pbv_reveng_20260712) |
| Find people | raw detector sees >=4 players on 96.7%/78.1% of frames (Burl/Wolv) | — | deep review S2 (research_deepreview_20260710) |
| Track people 2D | mean IDF1 0.852, worst 0.756 (broad internal); margin candidate: worst 0.6425->0.8516, cov4 0.0433->0.7117 (2-clip, PENDING, license OK'd internal) | 29.8-32.3 fps (det+track control) | full bar = IDF1>=0.85 + cov>=0.95 on FRESH clips (trk_reid_apron_20260712) |
| People 3D (BODY) | root-rel 59.7mm / PA 39.9mm (external bench); decode chain 23.4mm p95; foot-slide OPEN: skeleton-direct 20.8-48.4mm fails <=30mm on 3/4 (old proxy passed) | BODY 285.7s cold / 248.6s warm (Jul-12 H100); −17% worker candidate default-off | gate = court-frame world-MPJPE <=50mm on independent GT (ns014, w4_footattr_fix) |
| Court | ours 6.61px vs pb.vision 5.67px median (frozen GT-free M4, Wolverine); M1 3.01/6.22px med/p90 | calibration 1.33s/clip; temporal-lock candidate 16ms/frame | auto-FIND learned PCK@5 still 0 on owner viewpoints — diversity wall = asks 1-2 (court_precision_harness_20260712) |
| Paddle | rectangle IoU 0.22-0.33, preview only | paddle stage 0.39s/clip | no pose/contact GT; ask 6 unlocks (paddlewire_p31_20260709) |
| E2E speed | — | current spine 523s cold / 487s warm (Jul-12 H100); historical best 379.5s (rev-11) not current-reproducible | ~122s of wall = intended refined-arc solve; top NS-06 lever = reuse-aware solve (world_perf_20260712) |

## Running right now
BL-E ball anchor-evidence fusion (overnight, verdict pending — early signal: honest kill) ·
docreview_20260713 (read-only doc audit). Fleet EMPTY, $0/hr.

## Money (07-12 window)
trkA 1.655h ~$2-3.5 + bodyC 1.37h ~$0.8-5.8, both deleted+list-confirmed. Total ~$3-9.
