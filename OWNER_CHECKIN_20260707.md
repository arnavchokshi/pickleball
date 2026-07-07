# OWNER CHECK-IN — 2026-07-07 (wave 3 close)

## STOPs: none open. One action when convenient:
**Push grant is text-only.** CLAUDE.md says pushes are granted via settings.json, but settings.json
only encodes add/commit — the permission classifier (correctly) refused my push. One-liner fix:
add `"Bash(git push *)"` to `.claude/settings.json` → `permissions.allow`, or run
`! git push origin main` yourself. Everything is committed locally; only transport is pending.

## The headline
**The foot-slide gate is GREEN on all four clips** — wave-2's only carried blocker is closed:
burlington 40.6→20.25mm, outdoor 56.0→22.50mm, wolverine 17.98, img1605 16.66 (bar 30mm, thresholds
frozen; p95 <12mm everywhere; zero blockers; root jumps 0/0/0/0; body gate 4/4). It took three
adversarial-verify rounds to get here honestly — round 2 caught a fix that made the gate unfailable
by construction; what shipped instead fails closed and reports a companion candidate-slide metric so
exclusions can never hide. Fresh-run candidates measure 17-23mm — clean, not defined-away.

## Your labels changed two decisions in one night
- Your 480 review frames became the measuring stick that **killed a bad plan**: the 2D-gated teacher
  sets score F1 0.395 vs raw WASB's 0.680 against your boxes — the gates cut ~60% of true balls to
  gain +0.03 precision. Raw single-WASB is now the blessed SST seed (the fusion ban stands); gated
  teachers are archived as a measured negative; physics-gated teaching waits for court auto-cal.
- Labels/hour datum booked: ~240 frames/hr incl. tooling friction.

## Your quota unlock, actioned within the hour
First fleet H100 validated (driver/CUDA/torch clean, ~8 steps/s ≈ 2.3× A100-40 on identical work).
P1-1 warm-start trained on it: internal-val F1@20px **0.0615 → 0.6104 (~10×)**, median error
167.9px → **2.73px**, precision@20 0.848. Recall 0.477 is the known gap — P1-2 fine-tuning on your
in-domain labels is the designed fix. Internal-val only; VERIFIED=0 untouched. H100-80GB spot is now
the default heavy worker for training; BODY-stage jobs stay on A100 until a separate compat check.

## Ready for you (both ~minutes)
- **CVAT is live** at localhost:8080 (guide: runs/lanes/w3_labelfactory_20260707/OWNER_LABELING_GUIDE.md).
- **Court-keypoint pass**: 6 frames (one per legal harvest source), ~8 min, package + import script at
  cvat_upload/court_keypoints_20260707/. Note: your "8 sources" self-corrected to 6 — the other two
  ARE the held-out reservations and were excluded rather than leaked.

## The one carried red + honest notes
- img1605's camera-motion AUTO probe scored 0.329 in-pipeline vs 53.7 in the lane's offline
  acceptance (~160× apart — a probe-context bug, wave-4 #1). It ran uncompensated and still passed
  every gate, so nothing is blocked; the stage remains available via --enable-camera-motion.
- grounding_refine is honestly inert (the new phase producer emits 0 confident phases on the eval
  clips). Its un-kill path is upstream per-foot evidence — wave-4 #2.
- Mid-wave disk cleanup deleted 34 prelabel sidecars AND their rally clips; everything was
  regenerated from sha-verified sources with reconstruction notes. If a future cleanup targets
  data/, flag it and I'll regenerate proactively.
- Fleet: all VMs down (fleet1 STOPPED w/ disk + a READY snapshot as the fan template; fan1 + both
  trainer VMs DELETED, list-confirmed). Wave GPU spend ~$9-13 total.

## Wave-4 queue (booked in BUILD_CHECKLIST closeout bullet)
1. cammotion probe-context diagnosis · 2. upstream foot-attribution (un-kill refine) ·
3. P1-2 fine-tune on your labels + threshold sweep · 4. P4 court auto-cal (unlocks physics teacher) ·
5. burlington mesh-vertices notice · 6. fleet IP/known_hosts protocol · 7. H100 BODY compat.
