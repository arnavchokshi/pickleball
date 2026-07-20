# Owner check-in — THE single always-current file (updated 2026-07-20, end of autonomous push)

⭐ HEADLINE: You asked for REAL results, not infrastructure. The push delivered TWO measured wins,
one staged user-visible proof (blocked only on your gcloud login), and four honest negatives that
each name their exact defect. Total GPU spend ~$17.5 of the $35 cap; fleet EMPTY, all VMs torn
down + list-confirmed. VERIFIED=0 everywhere (nothing promoted — correct).

## ⚡ THE ONE THING TO DO NOW: type `! gcloud auth login`
Auth expired mid-push (fleet was already safely empty). The moment you re-auth, I fire the staged
GPU re-run that is the night's biggest user-visible proof: the Drill-clip replay WITH pooled
calibration → if the gate opens as the diagnostic predicts, PLAYERS + MESHES APPEAR ON A FRESH
CLIP for the first time. Command staged: runs/lanes/pooling_wire_20260720/RERUN_CMD.md (~$2-3, 45min).

## REAL RESULTS (measured, frozen scorers)
| Track | Number that moved | Meaning |
|---|---|---|
| ⚽ BALL 2D→3D | Event head 0 preds/0 TP → **510 preds, 70 TP@±2f, F1 0.215** (0.329@±5f) on the frozen 50-clip gate | The weighted-loss fix BROKE the all-negative collapse — the training machinery provably learns. Not yet a usable detector (low precision, tennis-domain); next lever = in-domain pickleball A/B/C |
| 🏟️ COURT | Gate-blocking far_centerline on the REAL Drill clip: **4/96 frames → 63 support, 0.357px, calibration counterfactually READY** | Your static-camera averaging intuition PROVEN on real footage. Wired default-OFF (ultra review in flight); the GPU re-run turns it into "players appear on fresh clips" |
| 🧾 pb.vision | 12 videos harvested + full-usage rights active; **~3,100 in-domain pickleball events** built into a training corpus (plausible 0.48-0.83 ev/s, 0 degenerate); frozen-stack replay on a fresh clip verified honest (82% 2D ball, viewer real, nothing fabricated) | The in-domain data the ball head was starved for + a 12x benchmark + the diagnosis that led to the court win |
| 🧍 PEOPLE | P0-I fabrication-prevention LANDED on main (3 ultra rounds): bridges refused, honest unbound export, provenance survives | Trust fix real; the ghost-CARD did not pass (below) |

## HONEST NEGATIVES (each with its named defect — none hidden)
1. **Ghost card NOT passed:** export-shape defect found+fixed ($0.30 GPU run), but the smoke shows
   Wolverine's bound slots themselves score 47 spectFP / cov4 0.24 — slot-binding quality is far
   below the design's own counterfactual (0 FP / 0.62 achievable on the same inputs). Named
   next-session target with measured headroom. No re-run burned on a known-fail.
2. **Court 5-venue harness: zero improvement** from all hardening levers (those venues fail for
   different reasons than missing-line recovery). The Drill-clip win is real; the harness stays honest.
3. **Cal k1 fix: doesn't beat raw** on the demo camera → in/out still abstains (reverted; capture
   discipline is the real lever).
4. **Fine-tune contract: DO_NOT_USE** from ultra review (a legacy code route bypasses the new
   64-frame asserts; stale-schema paths reachable) — exact file:line fixes recorded; blocks the
   pb.vision A/B/C until fixed (next session's first lane).

## COMMITTED TO MAIN TODAY (all ultra-reviewed)
P0-I selection layer + unbound-export separation; RF-DETR PENDING/non-default + detector-aware
cache identity (+ a pre-existing best_stack schema bug fixed); event-head loss fix + scale-up code;
research syntheses + advisory + all rulings/evidence. Plus: pb.vision full-usage ruling in North
Star §2.3; the pooling wire commits after its ultra review clears.

## NEXT SESSION QUEUE (in order)
1. (You, 30s) `gcloud auth login` → I fire the pooled-calibration replay re-run (the visible win).
2. Fix the fine-tune contract blockers (file:line list ready) → run the pb.vision A/B/C
   (owner-only vs teacher vs placebo) from the T20 checkpoint = the ball domain-gap experiment.
3. Fix Wolverine slot-binding to the design counterfactual → the one-shot ghost card.
4. RF-DETR GPU reproduction (flip gate) — cheap, unlocks the coverage win on burlington.
5. MonoTrack rally-DP (zero-label +16pp recall lever) once the event head fires plausibly.

## MONEY
~$17.5 GPU today (retrain $4.6-6.2 + replay $2-3.3 + card $0.3 + staging $1.3-1.8 + misc). Fleet
EMPTY, confirmed. All ledgers current (gpu_fleet.md).
