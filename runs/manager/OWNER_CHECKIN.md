# Owner check-in — THE single always-current file (updated 2026-07-13 ~01:0x, doc/org session)

⭐ HEADLINE: Docs reconciled + repo reorganized after the 07-12 sprint/court windows; a fresh-clone
break was found+fixed (stranded schema hunk). Fleet EMPTY (~$3-9 spent 07-12). One ball lane
(anchor-evidence fusion, the last executable ball-lift path) finishing overnight — early signal is
an honest kill, which would make further ball-3D gains YOUR-LABEL-GATED. VERIFIED=0 unchanged.

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
6. **Gold capture half-day (standing):** paddle 6DoF, ball-3D depth, and BODY accuracy are all
   capped until this exists. `runs/lanes/ns021_goldcapture_20260709/OWNER_HALF_DAY_CHECKLIST.md`.

## Best results so far (accuracy + time; ALL internal/protocol-caveated, none promotion-gated)
| Capability | Best accuracy | Best time | Note (evidence under runs/) |
|---|---|---|---|
| Ball 2D | F1@20 0.7248, recall 0.626, hFP 0.063 (zero-shot anchor, held-out Outdoor) | detector ~0.5-1.1x realtime (A100) | label curve alive: 1k-label ckpt 0.6152 internal; gate F1>=0.90 (heldout_eval_ledger row 4) |
| Ball 3D | 100% physics-plausible emitted; rally coverage 58/252 vs pb.vision 183/252 (Wolverine) | inside world stage (~122s incl. refined arc solve) | wall = upstream 2D/event evidence, not the solver (research_pbv_reveng_20260712) |
| Find people | raw detector sees >=4 players on 96.7%/78.1% of frames (Burl/Wolv) | — | deep review S2 (research_deepreview_20260710) |
| Track people 2D | mean IDF1 0.852, worst 0.756 (broad internal); margin candidate: worst 0.6425->0.8516, cov4 0.0433->0.7117 (2-clip, PENDING, license OK'd internal) | — | full bar = IDF1>=0.85 + cov>=0.95 on FRESH clips (trk_reid_apron_20260712) |
| People 3D (BODY) | root-rel 59.7mm / PA 39.9mm (external bench); decode chain 23.4mm p95; foot-slide 0.017-0.023m all 4 clips (gate <=0.03) | BODY 307s H100/Wolverine; persistent-worker −17% candidate default-off | gate = court-frame world-MPJPE <=50mm on independent GT (ns014, w4_footattr_fix) |
| Court | ours 6.61px vs pb.vision 5.67px median (frozen GT-free M4, Wolverine); M1 3.01/6.22px med/p90 | ~seconds/clip; temporal-lock candidate 16ms/frame | auto-FIND learned PCK@5 still 0 on owner viewpoints — diversity wall = asks 1-2 (court_precision_harness_20260712) |
| Paddle | rectangle IoU 0.22-0.33, preview only | — | no pose/contact GT; ask 6 unlocks (paddlewire_p31_20260709) |
| E2E speed | — | 379.5s full stack Wolverine (H100, rev-11 best); 489.4s ±1.6% x6 (rev-9 mean) | ~122s of wall = intended refined-arc work; top NS-06 lever = reuse-aware solve (world_perf_20260712) |

## Running right now
BL-E ball anchor-evidence fusion (overnight, verdict pending — early signal: honest kill) ·
docreview_20260713 (read-only doc audit). Fleet EMPTY, $0/hr.

## Money (07-12 window)
trkA 1.655h ~$2-3.5 + bodyC 1.37h ~$0.8-5.8, both deleted+list-confirmed. Total ~$3-9.
