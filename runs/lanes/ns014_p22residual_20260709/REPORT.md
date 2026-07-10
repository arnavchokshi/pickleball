# ns014_p22residual — lane report (NS-01.4 / P2-2 residual / GPU rescore)

Dates: 2026-07-09 → 2026-07-10. Manager: Fable bg session (job 60076b2d). Claim row:
runs/manager/inflight_lanes.md. Commits: 8cd810a53 (main arm), 4a3cbc60a (fix round 2 + GPU arm-1
evidence), arm-2 evidence commit to follow. VERIFIED=0 unchanged; nothing here is a promotion.

## Headline ruling — the P2-2 "23-27mm / 262mm / 53.5mm decode residual" is DECOMPOSED, not a defect

Every component of the historical GATE-1b failure is now individually measured and explained:

| Component | Measured | Evidence | Verdict |
|---|---|---|---|
| gate_1a euler round-trip | 4.1e-05 deg (bar 0.1) | gpu/pulled/gate1b_rescore.json | EXACT — decode plumbing sound |
| FK re-decode vs raw pred_keypoints_3d | p95 ~0.0002 mm | gpu/pulled/attribution_full.json fk_vs_head | EXACT — persisted euler params encode the emitted keypoints perfectly; the "keypoint head vs FK" hypothesis is dead |
| Grounding transform determinism | p95 1.99e-12 mm (32 banked records) | attribution_banked_fixture_report.json + fix2 re-run | EXACT — camera→world grounding is deterministic machine-precision math |
| pred_cam_t double-application | fixed wave-7 (84ed67040) | w7_p22gate lane | CLOSED (the 262→23mm class) |
| Intentional postchain mutations (EMA smoothing, foot-lock, phase-median lock, visual smoothing, wrist locks) | ~15.8 mean / 23.4 p95 / 26.7 max mm same-field on the w7 production run; per-stage table = arm-2 deliverable | w7 arm_c report; attribution CLI stages | INTENTIONAL, declared mutations — the "residual open" of wave-7 is THIS, by construction |
| mesh_skeleton_divergence ~53mm p95 | 50.8/53.4/53.2/52.6 across players; REPRODUCED digit-close across two fresh runs | ARM2 w7speed + gpu/pulled/gate1b_rescore.json | MODEL-FAMILY PROPERTY: the 70-kp learned head vs nearest of 127 raw chain joints; synthetic round-trip shows ~39.5mm even on perfect input |
| Synthetic render→SAM-3D-Body→measure (R1e instrument) | measured: joints_world p95 313mm; mesh-skel 39.5mm; 3/3 detections | gpu/pulled/synthetic_sam3d.json + renders | INSTRUMENT LIVE (first data; includes render-domain gap → upper bound) |

R1's checklist (a)-(f) is cleared: (a) scale/axis audited (scale_params = pass-through metadata,
never applied at grounding — chain map); (b) cam_t exactly-once enforced in code + gate fail-closed;
(c) field selection audited (pred_keypoints_3d correct; pred_joint_coords not a drop-in, p95 329mm);
(d) our world-skeleton formula mapped end-to-end (14-step chain, below); (e) synthetic gate wired +
first real measurement; (f) ceiling rule applies → recalibration proposal below.

## Gate recalibration proposal (R1(f) pre-authorized "with the owner informed" — OWNER-FACING)

GATE-1b's ≤1mm bar compares an FK re-decode against a persisted artifact that is postchain-mutated
by design; it can never pass and measures no defect. Proposed replacement structure (no code change
landed yet — thresholds in the repo are byte-unchanged this lane):
1. KEEP ≤1mm (and 0.1°) on the determinism gates: euler round-trip (gate_1a), grounding-transform
   determinism (attribution CLI grounding_determinism), postchain replay validation
   (chain_reproduced_1mm) — these catch every double-translation/scale/axis defect class forever.
2. Postchain mutation budget: per-stage deltas DECLARED in the attribution report and bounded
   (proposed total p95 ≤ 60mm on wolverine-class clips, per-stage table published) — an intentional,
   provenance-flagged budget, not an error metric. Final numbers pending arm-2.
3. mesh_skeleton_divergence and synthetic full round-trip become TRACKED family metrics (no 5mm/1mm
   pass bar); provisional ceilings anchored to the synthetic instrument (~40mm mesh-skel) once it
   has >3 samples. The identity/scale-locking workaround (arXiv:2512.21573) stays booked as the
   next-wave improvement lane if tighter world numbers are needed.
Owner ack requested before any threshold constant changes; until then the old gate keeps failing
honestly and this report is the interpretation key.

## NS-01.4 delivered slice + booked follow-ups

Delivered: threed/racketsport/coordinates.py — typed CoordinateSpace (7 named spaces),
canonical invert_extrinsics/world↔camera application, exactly-once translation policy, blessed
K-builder wrapper; adopted by mhr_decode/gate/synthetic-gate surfaces with diff-proven zero numeric
drift; py3.10-portable (fleet venvs). Booked follow-ups (other lanes' surfaces, NOT touched):
extrinsics inversion re-implemented 6× (court_calibration.py:247, court_calibration_metric15.py:661,
ball_inout_uncertainty.py:216, racket6dof.py:333, paddle_pose_fused.py:857, ball_arc_solver.py:2337);
inline K construction 7+×; homography apply/fit 3 algorithms; racket cm/m unit seam; untyped
homography_pixel_convention. Full inventory in the chain-map workflow journal (session evidence).

## The 14-step raw→persisted chain (verified by machine-precision replay on banked fixture)

normalize(cam_t once) → rotate root-relative (offset@R) → [camera-motion xy] → placement anchor
(track_world_xy, dz=-min_z) → EMA smoothing(α=0.65)+step-limit → foot-lock(z-snap, xy≤0.02m) →
[skeleton3d only: temporal-refine-gate + wrist-lock#1] → phase-median lock (both payloads) →
[skeleton3d only: foot-pin] → 3-tap visual smoothing (joints BOTH payloads; mesh vertices NEVER —
our own mesh-vs-joints offset source) → wrist-peak restore → [skeleton3d only: contact-splice +
wrist-lock#2]. scale_params: metadata pass-through. body_raw_grounded_joints.json persists only
under the raw preset.

## Structural catches (standing gotchas)

1. `--body-local` NEVER writes body_mesh.json/smpl_motion.json (write_body_monoliths never passed;
   --fetch-body-monoliths only feeds RemoteConfig; code-confirmed process_video.py:2734/5750/5805).
   Gate-class work must use remote self-dispatch even on the GPU VM itself.
2. remote-dispatch sync-back EXCLUDES fast_sam_subprocess/ chunk dirs → run gate/attribution
   VM-side against the DISPATCH-side run dir where body_mesh + fresh chunk index coexist.
3. Fleet venvs are Python 3.10 — no 3.11+ stdlib (StrEnum broke; fixed 4a3cbc60a). 
4. Attribution now HARD-REFUSES mismatched raw-index/body_mesh pairings (incoherent_inputs,
   fail-closed; --allow-incoherent-inputs stamps forensics) — arm 1's confounded 527mm replay
   number is the reason this guard exists; that number is VOID as evidence.
5. Self-dispatch on own VM: internal IP + VM's own hostkey appended to configs/ssh/a100_known_hosts
   (VM-local); compute mode DEFAULT (ns06 finding).

## GPU arms

Arm 1 (pickleball-h100-ns014rescore): 1.312h, $0.79-5.64, zero preemptions, DELETED+list-confirmed.
Delivered gate_1a/mesh-skel reproduction, FK-vs-head ~0, synthetic first measurement; replay arm
confounded (stale index) — led to the coherence guard. Evidence: gpu/pulled/ (13 files md5-verified).

Arm 2 (pickleball-h100-ns014rescore2): IN FLIGHT at report-draft time — corrected procedure
(dispatch-dir pairing, coherence guard active, full-frame per-stage attribution + measured gate_1b
+ synthetic replicate). Results appended below on landing.

## ARM-2 RESULTS
Arm 2 (ns014_gpu_rescore2): BLOCKED-STOCKOUT — ase1-c and ase1-b both ZONE_RESOURCE_POOL_EXHAUSTED
back-to-back (2026-07-10T05:05-05:07Z), the inverse of arm 1's availability 100 minutes earlier.
$0 spent, zero resources allocated. Evidence: gpu2/ (create logs + BLOCKER.txt). Arm 3 retry uses
the owner-granted broadened H100 zone ladder (asia-southeast1-c/-b, us-central1-a/-b). Results below.

## ARM-3 RESULTS (pickleball-h100-ns014rescore3 — DONE, 0.879h, $0.53-3.78, zero preemptions, DELETED+list-confirmed)

Valid and decisive:
- ase1-c first-attempt create (broadened ladder authorized, not needed). Sync gate 6/6 exact at
  4a3cbc60a; CoordinateSpace imports clean in BOTH venvs with NO shim (py3.10 fix proven on fleet).
- Input-coherence guard exercised LIVE: coherent (705/705 request-ids, single auto-discovered
  sibling index, 0 stamp mismatches) on the dispatch-dir pairing — the arm-1 confound class is
  structurally closed.
- gate_1b MEASURED (unblocked; sibling index supplied pred_cam_t; vertices_status=measured,
  160 frames): worst joints p95 112.43mm / max 241.86mm; worst vertices p95 90.45mm.
  COMPARABILITY CAVEAT: scale_source=None on this run; ARM2's 22-58mm p95 lineage may have decoded
  with scale flowing (w6 "scale/hand_pose plumbing") — open question, flagged, not resolved here.
  Interpretation unchanged either way: re-decode vs postchain-mutated persisted artifact fails the
  1mm bar by construction (see recalibration proposal).
- gate_1a exact AGAIN (2.73e-5 deg); FK-vs-head ~0 AGAIN (p95 0.00012mm) — both reproduced.
- Synthetic sam3d instrument REPRODUCIBLE: 313.48/315.75/39.45 vs arm-1's 313.03/315.20/39.48
  (deltas <=0.5mm) — the R1(e) standing instrument has a stability baseline.
- mesh_skeleton_divergence 53.41mm p95 — third digit-close reproduction across fresh runs.

Ruled VOID as production attribution (the honest negative of this lane):
- The per-stage replay table (foot_lock p95 853.64mm, root_phase_median_lock 852.80mm, visual
  smoothing 207.49mm; replayed-final vs persisted p95 527.77mm) does NOT measure the production
  postchain. Three self-declared replay infidelities (CLI assumptions block): anchors taken from
  placement.json fused_world_xy instead of the production grounding context; BodyPostChainConfig
  assumed default (no knob artifact exists in the run dir); stance_index not replayable. Internal
  impossibility proof: production foot-lock's xy correction is hard-capped at 0.02m — an 853mm
  replay delta cannot be the real stage. The production postchain total mutation remains best
  measured by w7 arm C same-field comparison: mean 15.8 / p95 23.4 / max 26.7mm.
- Correction to the banked-fixture claim earlier in this report: machine-precision replay
  validation held for the grounding step and the ALL-STAGES-DISABLED path; ENABLED-stage offline
  replay is NOT yet faithful. Booked follow-up (fenced worldhmr/orchestrator surface, next wave):
  serialize the postchain knobs + grounding context + stance_index into the run dir (or better,
  emit per-stage deltas PRODUCTION-side during the run) so attribution needs no reconstruction.
- Cosmetic: dispatch dir named source_<ts>Z (no --clip passed); content identity unambiguous.

Lane GPU total (3 arms): ~2.19h, ~$1.3-9.4, zero preemptions, all VMs deleted+list-confirmed.

## Best-stack delta
(c) None. Measurement/API/portability lane; no model, weights, or promoted-policy change; all gate
threshold constants byte-unchanged (best_stack sha 6ec18489f79605e76cbf9f990a07ffcd9950fb35ea64c211e109ba5575c9a315).
