# Owner check-in — 2026-07-12 (12-14h autonomous sprint)

⭐ HEADLINE: Sprint running. The 3 tranche-1 lanes killed by your laptop sleeping on 07-10 night
(ballcand / spine017 / coords_parity) were resumed with full context and are landing; orphaned
labeling-session work (scratch-mode CVAT import + provenance ingest) adjudicated + committed +
pushed. New product lanes + GPU attestation waves dispatching through the day per
runs/lanes/plan_sprint_20260712/PLAN.md. caffeinate armed so sleep can't kill lanes again.

## Blockers
(none so far)

## Your asks when back (each ≤15 min, easiest form — numbered, do in any order)
1. COURT (the highest-value 1h you can give the model): start Docker Desktop + the CVAT stack, then run
   `.venv/bin/python cvat_upload/court_diversity_20260712/import_court_diversity_tasks.py` and label the
   4 shards (25 frames each, ~15 min/shard) per cvat_upload/court_diversity_20260712/OWNER_GUIDE.md.
   100 frames from 28 NEW venues — this attacks the exact wall that killed the last court wave.
2. BALL: finish labeling CVAT task 87 (350-frame uniform scratch audit) whenever — ingest is committed
   and ready with per-row provenance.
3. (more staged at session close)

## Money / GPU log
- Fleet at sprint start: EMPTY (fleet1 TERMINATED only). gcloud auth was already live — no action needed.
- (per-VM rows appended as provisioned)

## Quick wins landed
- Scratch/audit-stratum labeling mode committed+pushed (328771272) — your task-87 labels can now be
  ingested with per-row provenance the moment you finish them.
- Viewer UX wave-2 committed+pushed (006810cb7): seekable event markers, camera presets, full entity
  toggles, trust pills, honest empty states — 255/255 tests, live-browser pass running.
- Your pb.vision directive is executing: reverse-engineering lane live on their cv export (method
  forensics + gap decomposition + a harness that scores every future candidate against them).
- iOS five-tab truth pass implemented (fabricated Stats placeholders KILLED, audited-facts-or-honest-
  empty wiring); simulator test execution running before commit.

## Evening headlines
- TRK "missing people" symptom: the GPU sweep found a big honest win-candidate — court-margin 1.0m +
  restored OSNet takes worst-clip IDF1 0.64->0.85 and four-player coverage 0.04->0.71 with ZERO new
  identity switches (internal cards; not promoted — full bar needs cov>=0.95 + a ReID license ruling).
- TT3D anchor search: pre-registered kill fired (fallback 9/13 vs <5/11 bar) — rejected honestly,
  DP follow-on killed, ball E2E GPU slot cancelled (saved an H100). Ball program pivoted to
  recovery-policy coverage (running) + size/depth cue (running).
- P0-C closed: unfingerprinted artifacts can no longer sneak through reuse — every future score is
  provenance-safe.
- BODY speed levers (persistent worker / compile cache) running on a second H100 now.

## Afternoon landings (all committed+pushed, each locally re-verified before commit)
- Tranche-1 recovered end-to-end: ball UKF/RANSAC candidates (both honest non-wins, flags off),
  spine runner honesty + refined-events + dependency hashes, typed-coordinates parity x2 slices.
- pb.vision reverse-engineering: their method decoded (event-anchored piecewise-ballistic), our gap
  quantified (emission policy + anchoring, NOT camera), comparison harness committed — TT3D
  integration lane running against it now.
- Timebase contract core (P0-D/P0-H) + coaching-facts core w/ zero-fabrication audit (NS-05.1).
- Viewer: UX wave-2 + 2 browser-found defect fixes (proportional scrub restored).
- iOS: five-tab truth pass, fabricated stats killed, 6 new tests green on simulator.
- TRK ReID/apron GPU sweep running on H100 (first-attempt create, frozen margin matrix).
- A concurrent court-precision session (your other window) is running with clean mutual fences.

## Overnight log
- ~10:30 session start: reconcile, 3 lane resumes + plan consult dispatched (all gpt-5.6-sol xhigh).
- ~10:45 orphan adoption commit 328771272 pushed; webux2 viewer UX lane dispatched.
- ~11:30 sol sprint plan landed+consumed; timebase/facts/iosUI lanes + court harvest tightening out.
- ~12:00 your pb.vision directive received -> pbv_reveng lane dispatched.
- ~12:10 webux2 PASS committed 006810cb7; browser + iOS-simulator verification lanes out.

---

## COURT-PRECISION session (Fable bg 40bcb767, started ~11:25 PDT) — your court directive

⭐ CVAT IS UP with your 100 diverse court frames READY TO LABEL: project
`racketsport_court_diversity_20260712`, tasks 88-91 (4 shards x 25 frames, ~10-15 min each,
~45-60 min total). http://localhost:8080 — place the 4 required corner points when visible,
optional points only when unambiguous, never guess (guide: cvat_upload/court_diversity_20260712/
OWNER_GUIDE.md). This labeling is THE unlock for learned court auto-find (court-wave verdict:
viewpoint diversity is the binding constraint). Import needed a cvat_sdk `mutable` attribute fix
(the harvest lane's flagged residual risk — real, now fixed; report:
import_report_20260712_courtsession.json).

Research landed (committed 9a4eba44b, runs/research_courtlock_20260712/): 58-agent fanout +
sol-xhigh consult, 9-rank plan. Headlines: our "refinement" was an unwired stub (confirmed);
pb.vision's court lock is mostly a STATIC-camera capture-constraint UX (CourtFocus pre-lock),
not moving-camera magic; direct pose optimization against line evidence beats
homography-then-refine (TVCalib ablation); mm-level in/out = multi-camera/laser rigs only, so
our honest product edge is covariance + abstention bands; 1px image error ~= 12.6-17.3cm on
court at current calibration (why precision work matters).

In flight when this note was written: GT-free court-precision harness (baselines: Wolverine M1
5.33px median @77.5% evidence coverage; per-frame temporal calibration DOESN'T EXIST YET —
that's the moving-camera gap), rank-1 guarded point+line pose optimizer, rank-2 subpixel
paint-centerline evidence. Section will be updated at window close.

### Court session close (~15:3x PDT) — what landed while you were out

1. **Label the 100 court frames when you can** (CVAT is up, tasks 88-91, ~45-60 min total). This is
   the single unlock for learned court auto-find. Everything else court-side is now waiting on it.
2. Committed (02d9acedf): frozen GT-free court-precision harness (our court vs pb.vision on the same
   frozen protocol: **ours 6.61px vs their 5.67px median** on Wolverine — the gap is now a number,
   not a feeling); hybrid subpixel paint-line evidence (default-off, feeds every future consumer);
   per-frame temporal court-lock candidate (the moving-camera piece nothing had); guarded optimizer
   replacing the dead refinement stub.
3. Honest kills you should know about: wholesale evidence-swap and aggressive homography refinement
   both died on measured evidence (3 adversarial rounds). The remaining ~1px gap to pb.vision lives
   in SEED quality (their static-camera CourtFocus capture lock) — which strengthens the case for
   our profile + guided-confirm + capture-guidance route, and for your labeling session.
4. To prove the temporal lock on real motion we need one reviewed MOVING owner clip (handheld pan
   of a rally, ~30s). If you record one, drop it in the usual import path and say the word.

