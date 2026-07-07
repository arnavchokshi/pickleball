# LANE w4_docs_20260707 — W4-F docs reconciliation (evidence-complete items ONLY)

## HARD RULES
No git branches/commits/pushes. Read NORTH_STAR_ROADMAP.md PART VI.2 + the last ~10 BUILD_CHECKLIST
bullets first. The 4 protected clips are EVAL-ONLY (you touch no data). Honest reporting: a doc may
never claim more than its linked evidence shows; VERIFIED=0 stands everywhere. Run the doc-focused
blast-radius suite (below), not a hand-picked subset. No new root .md files. Artifacts under
runs/lanes/w4_docs_20260707/.

## EXPLICIT FILE OWNERSHIP (this lane ONLY)
CAPABILITIES.md, PIPELINE_STATUS.md, NORTH_STAR_ROADMAP.md, TECH_BLUEPRINTS.md,
FABLE_OPERATING_MANUAL.md. DO NOT TOUCH: BUILD_CHECKLIST.md (manager-owned closeout),
runs/manager/**, ios/**, scripts/**, threed/**, tests/** (except NONE — you edit docs only),
list_scaffold_tools.py, scripts/racketsport/doctor.py (foreign in-flight).

## OBJECTIVE
Reconcile the five docs with wave-4's LANDED, evidence-complete facts (NORTH_STAR Part IV rule 14).
GPU-proof-dependent lines get an explicit marker `[PENDING wave-4 decisive proof — manager fills at
closeout]`, never a claim. Every fact you write must cite its evidence (commit sha or
runs/lanes/... path). The landed facts (verify each against git log / the lane reports before
writing — state each as a CHECK):

1. Camera-motion (BODY pillar + CAPABILITIES): decode-orientation explicit policy + fail-safe
   mismatch semantics landed cd0b59390 after 2 adversarial verify rounds (img1605 production probe
   53.70515 AUTO ON; statics bit-exact); first-class summary keys landed 1588b110f (integration
   lane). Wave-3's "probe-context diagnosis = wave-4 #1" correction blocks are now RESOLVED —
   update the BODY pillar wave-3 corrections block accordingly. Decisive in-pipeline proof
   [PENDING].
2. Foot-attribution (BODY pillar): skeleton-direct per-foot producer landed UNWIRED 75e438223
   after 2 verify rounds; measured negative — honest phases 2-12/clip, predicted gate breaches
   3/4 clips (34.6/33.6/48.4mm vs frozen 30mm), dominant rejection phase_penetrates_ground
   120-167/clip ⇒ raw-skeleton noise is the binding constraint; grounding_refine stays honest
   no-op; un-kill re-queued behind P2-2 (wave 5). The forbidden gate-referencing exclusion also
   removed from body_grounding_quality.py in the same commit.
3. BVP (BALL-3D pillar STEP 1 status): 3-round arc landed 5633c4b48 as PARTIAL — LOO per-holdout
   refit REAL now (verifier-confirmed 5 unique param sets), junction-sanity helpers inactive,
   anchor-preservation diagnostics; D.3(b) span protection NOT achieved (r1 span-equivalence
   refuted by adversarial adjudication; r2 runtime-killed; r3 primary failed quality) — banked
   wave-5 design: frozen-baseline arc params as protected-span priors + junction repair before
   validity gates. STEP 2 (Magnus) stays gated behind STEP 1 per the mandatory ordering.
4. Court (COURT pillar + CAPABILITIES): harvest per-source calibration from owner court-kp labels
   landed 83e090168 — 1/6 sources manual_bar (73VurrTKCZ8 median 2.93px / p95 6.0px; 8/40 clips
   covered), 2 full-labeled sources FAIL p95 (36.2/32.2px, net/far-side residuals — owner relabel
   queued), physics-gated SST teacher stays DEFERRED (<2 sources at bar, kill honored);
   run_ball_chain --court-calibration handoff exists.
5. BALL-2D pillar: STEP-3 build gap CLOSED 5b268aa6d (train_ball_stage2.py sparse-review
   semantics: reviewed-only 486 rows = 268 pos + 218 reviewed-absent; occlusion-aug paired w/
   WBCE; SST manifest + disagreement CLIs; the dense CVAT helper documented as unsafe-for-sparse
   and bypassed). The LOCAL BLOCKER line (§1) is CLOSED: models/checkpoints/wasb/
   wasb_tennis_best.pth.tar now local, sha-verified (w4_ballgpu prestage). Stage-2 seed-tune +
   SST-r1 internal-val cards [PENDING — w4_ballgpu in flight].
6. Fleet (manual §12/§19 + gpu_fleet references): DEFAULT_REMOTE_HOST is REMOVED (landed
   dcc4dae42 — --remote-host required fail-loud; refresh_remote_host.sh helper) — fix the
   manual's stale narrative lines that still describe the constant as pending-update. H100 is now
   BODY-VALIDATED ≈2.37× A100 (w4_h100body: 479.6s vs 1134.4s wolverine BODY; snapshot→a3 boots
   with pd-balanced; evidence runs/lanes/w4_h100body_20260707/REPORT.md) — update the manual §19
   SKU guidance + TECH_BLUEPRINTS references: H100 default for training AND BODY within cap;
   decisive gate runs stay on proven SKUs. Version stamps now hash committed blobs (190dea09f).
7. Mesh telemetry (CAPABILITIES/PIPELINE_STATUS if they mention viewer warnings):
   missing_embedded_mesh_vertices vs true absence distinguished (684d03380).
8. NORTH_STAR PART III: update the STATUS lines for the tasks these landings touch (P1-4a partial,
   P2-1 resolved-pending-proof, P4 harvest-cal measured, P1-2 build-gap closed) with dated
   evidence pointers; PART VI: append a short VI.2 wave-4 execution log paragraph (what ran, what
   landed, what carried — cite the closeout bullet as [PENDING] for final numbers).

## EVIDENCE TO READ FIRST
git log --oneline 7fdbdb36e..HEAD (the wave-4 landings); runs/lanes/w4_*/report.json +
report_r2/r3 where present; runs/lanes/w4_h100body_20260707/REPORT.md;
runs/lanes/w4_footattr_fix_20260707/report_r2.json (the measured-negative numbers);
runs/manager/wave4_boot_prompt.md.

## ACCEPTANCE
- Every edited claim carries a dated evidence pointer (sha or path); every proof-dependent line
  carries the explicit [PENDING...] marker; zero claims beyond evidence.
- Doc blast radius green: `.venv/bin/python -m pytest tests/racketsport/test_truthful_capabilities.py -q`
  plus every doc-consistency/inventory test your grep finds (list them; run ALL; the markdown doc
  inventory test must pass — you create no new root docs).
- A diffstat per file in the report CHANGES.

## STRUCTURED REPORT
objective_result; acceptance table (per doc: claims added/corrected, pending markers count);
full_suite; honest_issues (anything you found contradicting the docs that you did NOT have
authority to resolve — list, don't fix); next.
