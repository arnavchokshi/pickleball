# Manager rulings — ball 2D->3D lift research (2026-07-09, owner priority directive)
Input: SYNTHESIS.md (39-agent sweep, 2-vote-refuted load-bearing claims marked [CORROBORATED]).
Key validation: pb.vision's own job posting confirms they run OUR architecture class (3D equations of
motion + drag + Magnus over imprecise CV output) — the gap is robustness engineering, not exotic tech.

## Adopt sequence (re-derived; each with the synthesis's kill criteria binding)
1. **SHIP FIRST (symptom killers, low effort):** fail-closed world-overlay fix (already booked from
   w7_ball3ddiag) + adopt-list #5 UKF-seeded graceful-degradation fallback (filterpy, seeded from
   adjacent well-fit segment) — converts fallback junk into plausible-or-hidden.
2. **CENTERPIECE (structural):** #1 TT3D-pattern joint anchor search — bounce state (ray-plane vs our
   court/net planes) becomes a FREE VARIABLE inside the existing scipy TRF+huber fit instead of a
   hard upstream gate. Directly attacks the 9/11 fallback rate. Kill: fallback must drop <5/11 on the
   diagnosed clip else the bottleneck is candidate density.
3. **CHEAP ANCHOR BOOSTERS (parallel micro-experiments):** #6 BlurBall blur-read from existing WASB
   heatmaps (sub-frame event timing, free signal); #7 visual-gates-audio ordering swap for contact
   windows. Both cheap to falsify.
4. **SOLVER HARDENING:** #2 both-ends segment pinning (MonoTrack) + #4 dedicated RANSAC inlier pass
   (Maiden patent pattern; splits "which detections to trust" from "where segments break").
5. **BIGGER, AFTER #1's KILL READS:** #3 whole-rally DP segmentation (TT3D).
## Owner decisions (surfaced, NOT unilateral)
- **Magnus/spin revisit (#8):** kill-list re-adjudication — NEW first-party evidence (pb.vision job
  posting names Magnus) satisfies the "only with new evidence" rule for a CONTROLLED experiment on
  the 2 well-fit segments; owner go required.
- **Learned lift-first architectures** (TT4D/Kienzle-WACV26/Where-Is-The-Ball): the field's own answer
  to our exact symptom, synthetic-training only (no 3D GT needed) — but a new-model-training decision;
  queue for owner if the physics path stalls after items 1-5.
## Owner action (unlocks the benchmark)
pb.vision in-app "Download Raw Data" (three-dot menu) -> "cv" JSON export for the SAME clips we
process: gives their per-frame numbers on our footage, no partner API needed. Then run the 3-pillar
no-GT protocol (reprojection consistency + court-plane bounce pseudo-GT + physics re-integration,
Bland-Altman framing) from SYNTHESIS.md §4.

## ADDENDUM (2026-07-09, owner supplied the pb.vision "cv" export — banked at pbvision_cv_export/)
Export contents (1 rally, 252 frames @30fps): per-frame 3D ball court_position with per-frame
`interpolated` flag + `selected` candidate (=they ship interpolation PROVENANCE — validates our
fail-closed/trust-band fix direction first-party); per-frame ball/bounce/net/shot confidences
(bounce peaks 0.97); `ball_radius` tracked per frame (= apparent-size DEPTH CUE — new adopt-list
item #9, cheap: WASB heatmap footprint size as a depth constraint in the arc fit); full camera solve
w/ per-court-point confidence+spread; player court positions (feet units, their frame).
BENCHMARK NOW CONCRETE: need the source clip identity from the owner -> run ours on the identical
clip -> 3-pillar protocol + Bland-Altman. Owner also reports labels ~3k => 3k checkpoint gate fires
next ball lane (train on new corpus revision; visibility ruling for w6+ sessions stands).

## ADDENDUM 2 (2026-07-09, head-to-head EXECUTED — runs/lanes/w7_pbv_compare_20260709/COMPARISON.md)
Alignment solid (r=0.949, ±1 frame). DECISIVE FACTS: our 2D coverage 80.6% vs PB 58.7% (overlap
agreement 5.7px p50); cameras agree 2.7cm/0.27deg (court solve NOT the differentiator); PB emits NO
3D on 31% of rally frames (fail-closed omission, interpolated flag used sparingly 0.58%) while we
emitted fallback garbage on ~88-91% — THE visible gap is emission POLICY; bounce landing pseudo-GT:
PB 0.22m vs ours 0.27m mean (n=2, comparable class). Open: 4.65m median 3D disagreement on the one
comparable well-fit segment — needs independent GT (gold capture), neither system is truth.
RULING CONFIRMED+STRENGTHENED: ship fail-closed omission + UKF fallback first (= instant PB-look
parity mechanism), then TT3D joint-anchor search to monetize our HIGHER 2D coverage. PB comparator
data + protocol now reusable for every future ball-3D change on this clip.
