# Owner check-in — THE single current one (updated 2026-07-13 ~00:1x)

⭐ HEADLINE: 12-14h sprint complete — ~20 commits pushed, all locally re-verified. The ball 2D->3D
question now has a precise, evidence-backed answer (see below). TRK "missing people" has a big
internal win-candidate ready to flip. All GPUs torn down (~$3-9 total today). One ball lane
(anchor-evidence fusion) re-running overnight; your other window's court-precision session ran
concurrently with clean fences.

## The ball 2D->3D answer (your directive)
pb.vision's export = our Wolverine clip, so we benchmarked head-to-head. Their method: event-anchored
piecewise-ballistic arcs + a learned ball-radius depth cue. We tried every reproduction path with
pre-registered kill bars; three killed honestly (TT3D anchors, aggressive gap-bridging, free size
proxy), one marginal survivor (conservative recovery v2, PENDING). NET: our lift's wall is upstream —
too few trustworthy 2D candidates/event anchors feeding the arc fits, NOT the solver/camera. The two
levers that move it: (1) anchor-evidence fusion (lane re-running now), (2) YOUR labels (asks 1-3) +
a learned radius head later. Every candidate now gets an instant pb.vision scorecard.

## Blockers
(none hard)

## Your asks (numbered, easiest-first, each self-contained)
1. **Court labels, ~1h, highest value:** start Docker Desktop, bring up CVAT as usual, then
   `.venv/bin/python cvat_upload/court_diversity_20260712/import_court_diversity_tasks.py`
   (dry-run already validated). Label the 4 shards (25 frames each) per
   `cvat_upload/court_diversity_20260712/OWNER_GUIDE.md`. 100 frames / 28 NEW venues — attacks the
   exact diversity wall that killed the court wave.
2. **Court tasks 88-91** (staged by your court-precision session) — label whenever; same CVAT stack.
3. **Ball task 87** (350-frame uniform scratch audit) — finish whenever; ingest is committed and
   provenance-aware.
4. **10-min phone fps test:** say the word and I stage a Wolverine replay URL for your iPhone —
   first-ever real-device fps number (all prior numbers were software-renderer artifacts).
5. **NS-01.2b physical proof (needs your signed device, ~15 min):** one 30s record -> upload -> open
   own replay trace on the real app.
6. **Gold capture half-day (standing):** paddle 6DoF + ball-3D + BODY accuracy are all capped until
   this exists. Checklist: `runs/lanes/ns021_goldcapture_20260709/OWNER_HALF_DAY_CHECKLIST.md`.

## Resolved by you (recorded)
- OSNet ReID license: internal/private use ruled fine (2026-07-13) — TRK margin candidate is now
  gate-blocked only (needs cov4 >=0.95 on fresh clips for full promotion; internal evidence is
  worst-clip IDF1 0.64->0.85, cov4 0.04->0.71, zero new switches). Can flip default-on as preview on
  your word.
- pb.vision export clip identity: RESOLVED — it is Wolverine (proven, frame-aligned).

## Money / GPU log (2026-07-12 sprint)
- pickleball-h100-trka: 1.655h, ~$2-3.5, deleted+list-confirmed (TRK sweep).
- pickleball-h100-bodyc: 1.37h, ~$0.8-5.8, deleted+list-confirmed (BODY levers — all honest-rejected;
  found the world-stage cost attribution instead).
- Fleet now: EMPTY (only old fleet1 disk, TERMINATED). Peak concurrency 2/5.

## Landed this sprint (all pushed, VERIFIED=0 unchanged)
Tranche-1 recovery (ball candidates / spine honesty / typed coords x2) · P0-C closed (stale reuse
dead) · NS-05.1 facts + zero-fabrication audit BUILT + runner-ENFORCED · P0-D/H timebase core ·
viewer UX wave-2 + 2 browser-found fixes (scrub restored) · iOS five-tab truth pass (fabricated
stats killed, sim-verified) · pb.vision forensics + harness · TRK margin PENDING candidate ·
ball recovery-v2 PENDING · size-obs sidecar + 3 honest ball negatives · court diversity pack staged ·
scratch-labeling mode adopted · world-stage 122s attributed (= intended refined-arc work; top speed
lever booked: reuse-aware solve).
