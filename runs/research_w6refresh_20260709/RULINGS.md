# Manager rulings — w6 deep review + North Star refresh (2026-07-09, Fable)
Inputs: internal_audit_synthesis.md (10 pillars + critic), sota_ball_synthesis.md, sota_body_synthesis.md
(all claims re-derived from the banked maps/verdicts; [CORROBORATED] = survived 2-vote refute, never VERIFIED).
Anchor (owner): the most accurate and fast processor from pickleball video -> 3D world, to give players
feedback on how to improve.

## R1 — P2-2 decode defect: from archaeology to a CHECKLIST lane (reshapes wave-7 #2)
The 262mm hunt is now enumerable: (a) audit our decode/placement against MHR conversion.py@4debaacf
L472-516 — the 100x cm/m scale must hit BOTH pred_vertices AND pred_cam_t at the same point; the
axis-flip ([:,[1,2]]*=-1) is branch-dependent; (b) verify pred_cam_t is added EXACTLY ONCE
[CORROBORATED — missing/double application reproduces our mesh-vs-skeleton divergence]; (c) check
WHICH field our harness reads as "skeleton" — pred_keypoints_3d lacks central spine joints vs the
127-joint pred_joint_coords (GH #34): 262mm worst-joint may be a field-definition mismatch; (d) there
is NO maintainer-blessed world-skeleton formula (conversion.py never reads joint fields) — our
placement is our own extrapolation, audit it as such; (e) build the SYNTHETIC render round-trip gate
(MetricHMSR recipe: authored mesh w/ known scale/pose/camera -> render -> SAM-3D-Body -> measure) as
the standing decode-fidelity instrument; (f) CEILING RULE: 53.5mm p95 is family-normal (MetricHMSR
3DPW PVE 62.7mm; anthropometric-fidelity audit 2601.06035 = regression-to-mean baked in) — if the
checklist clears and residual ~50mm p95 remains, STOP chasing and switch to the WORKAROUND: per-track
identity/scale locking + latent-space smoothing per arXiv:2512.21573 (published on OUR backbone;
also covers a soft foot-contact grounding alternative for the killed grounding_refine). GATE-1b's
<=1mm bar was mis-calibrated for this model family — recalibrate the gate to the synthetic-gate
result + locked-identity residual, with the owner informed.

## R2 — Label economics: gated, stratified, not flat (amends the 10-20k target)
(a) Checkpoint evals at 1k / 3k / 6k / 10k reviewed rows [CORROBORATED plateau evidence 2601.13380]:
if the curve flattens, surplus owner-hours go to venue/lighting DIVERSITY or coaching-wave labels,
not raw volume. (b) Add a small UNIFORM-RANDOM audit stratum alongside the disagreement queue (the
BINDING selection-bias caveat gets a structural fix; enables honest absolute numbers). (c) Held-out
ledger gains a SEEN-vs-UNSEEN environment split; the gap itself is a first-class metric
[CORROBORATED ~16pt gap is normal at our scale — recalibrates the 0.7248 wall narrative].
(d) Fold TOTNet-style occlusion augmentation into the next seed training (recipe, not architecture).
(e) Before the next fine-tune: cheap re-run of the 486-row seed LoSO anomaly (0.6404 vs control
0.6858) so the winning lineage is clean.

## R3 — Ball detector architecture: no change now
TrackNetV5/TOTNet/RF-DETR are same-domain benchmarks; nothing beats our zero-shot cross-domain setup
on evidence. Bench candidates ONLY after the label flywheel matures (post-3k checkpoint). Spin stays
killed (event-camera evidence corroborates the noise-floor read); TT3D bounce-kink spin is the future
revisit vector, gated as before on view-geometry confidence + now bounce-frame availability.

## R4 — Playback/streaming: byte-budget stands; MeshOpt when glTF; strategic watch on 4DGS
Keep the landed byte-budget (owner decides 300-vs-400 + human_review-tier display). If/when the
viewer moves to glTF/GLB: MeshOpt + KHR_mesh_quantization, NOT Draco (Draco mishandles per-frame
vertex animation). Calendar flags: MPEG V-DMC browser decoder (~post-2026-03 IS), 4DGS web streaming
as the alternative rendering target IF decode work stalls.

## R5 — Speed: safe lever = frame-selection/conditioning; certify before claiming
(a) P5-1 clean-room gate-scoring BEFORE any further speed claims (the 3.8x headline is uncertified
against its own gate; the one attempted lever regressed). (b) SAM-Body4D masklet conditioning = the
cheap decode-independent eval candidate (batching win w/o accuracy regression class). (c) The decode
fix may be speed-POSITIVE (~381ms/frame iterative fitting share in the family baseline) — profile our
stage during the R1 lane. (d) Full-model-rewrite speedups remain distrusted (NOT-ADOPT vindicated);
verify the 2603.15603 March-2026 paper is the same lineage we benched (30-sec check in the R1 lane).

## R6 — Priority resequencing (from the internal audit, evidence at each item)
(1) P3-1 paddle wiring = IMMEDIATE next-wave lane (4-wave-old BUILT-NOT-WIRED orphan; zero research
needed). (2) Browser-verify dev-bypass = small lane, stop carrying it. (3) P4-0 court-profile library
resequenced AHEAD of any 3rd auto-find retrain — owner re-confirms the override with the 244.3/212.6px
evidence in hand (their session's call; surfaced on the board). (4) NEW first-class items:
security/PII/secrets review BEFORE product flags flip live; training-data licensing check (GPL
PnLCalib + Roboflow ToS vs Stripe monetization); input-quality guardrail (P5-5b elevated — garbage-in
protection is product-critical). (5) Consolidated OWNER-TIME queue (labels 42-84h, paddle GT session,
phone tests, game recording, 4.0 reviewer, GCP invoice, mesh display ruling) — ranked in ONE place at
every wave boot instead of per-pillar re-surfacing. (6) Consolidated fleet-spend-vs-ask table at wave
boot (three pillars independently propose GPU hours).

## R7 — Docs truth-ups (all 8 stale items from the audit table get fixed in the refresh lane)
CAPABILITIES ball row (seed_official is the LoSO winner; stage1 below control) · CAPABILITIES body
row (GATE-1b legitimate FAIL numbers + meshcap win) · NORTH_STAR P0-4 corpus 486->1121 · P4-1 landed
(unblock stale note) · P4-2/P4-3 honest-miss numbers · PHASE-5 speed header softened to
gate-unscored · PHASE-7 Swift file count 140 · TECH_BLUEPRINTS paddle P3-1 resequencing note.
