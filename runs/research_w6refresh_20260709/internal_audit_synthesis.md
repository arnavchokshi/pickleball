# Progress Audit — 10-Pillar Synthesis for NORTH_STAR Refresh
*(sources: ball-chain, body-world, data-engine, court, paddle-racket, speed-pipeline, product-infra-viewer, coaching-readiness pillar maps + critic pass; 2026-07-08)*

---

## 1. WORKING WELL — compounding assets

**Validation/measurement discipline (meta-asset, cross-pillar)**
- LoSO harness validated twice — first on synthetic-legal folds (proved LoSO-mean beats pooled, abs-err 0.058 vs 0.074), then re-run on real owner outdoor labels with a true outdoor fold, confirming `seed_official` (raw WASB) as current winner (F1 0.5329 vs 0.3611/0.2971). `runs/lanes/ball_loso_validation_20260707/report.json`; `runs/lanes/w6_labelingest_20260708/gpu_rescore/loso/loso_report.json`
- Preprocessing train/inference mismatch found + fixed + retrain-validated (wave-4 Wolverine degenerate class 0.20→0.75 F1 after fix). `runs/lanes/w5_ballprep_20260707/report.json`; `runs/lanes/w5_ballretrain_20260707/vm_pull/`
- Two honest kills on real evidence, not guesswork: Magnus/BVP-spin (`fit_spin_scalar=False`, worse on 2/3 clips, `runs/lanes/w6_magnus_20260708/verify/reprojection_compare.json`); Fast-SAM-3D-Body (1.31x slower full-stage, +149mm/frame on fast swings, `BUILD_CHECKLIST.md:741`).
- GATE-1b harness itself was rebuilt after catching its own 2 real defects (crash on missing `artifact_size_policy`; silently-dropped `scale`/`hand_pose` fields) before trusting its FAIL result. `runs/lanes/w6_close_errand_20260708/gate1b_raw_arm_report.json`

**Data engine flywheel — LIVE end-to-end**
- Harvest→WASB prelabel→CVAT→owner export→deterministic ingest→LoSO rebuild proven byte-identical across reruns (md5 `37a5d43ab537a15bd12d382bb882a5fe`), protected-label scan 0 hits across 10 zips. `runs/lanes/w6_labelingest_20260708/report.json`; `runs/lanes/w6_labelpack_20260708/validation_report.json`
- Corpus 486→1121 rows in one wave with visibility taxonomy populated. `runs/lanes/w6_labelingest_20260708/corpus_md5_manifest.json`
- Corpus dashboard (P0-4) catches leakage in tests, 0 collisions on real data. `runs/lanes/p04_corpus_dashboard_20260707/report.json`

**Body-world**
- Fresh-GPU decisive proof 4/4 green (slide, root-jump, browser assertions). `runs/lanes/w4_freshproof_20260707/summary.json`
- GATE-1a idempotence passes cleanly (4.098e-05 vs 0.1 target). `runs/lanes/w6_close_errand_20260708/gate1b_raw_arm_report.json`
- Mesh byte-budget landed with real measured win once tier-eligibility accounted for (outdoor 4.1x fps). `BUILD_CHECKLIST.md` [W6 RULINGS 2026-07-09] RULING C; `runs/lanes/w6_meshcap_20260708/report.json`

**Court**
- Manual/metric-15pt path is production-real today (12.3px p95). `CAPABILITIES.md:97`; `NORTH_STAR_ROADMAP.md:1281-1290`
- P4-1 Wave-A solver actually landed on main (23/23 byte-equal, previously blocked). `BUILD_CHECKLIST.md:753`
- calv1_seeddiag correctly diagnosed domain gap with zero code spend before committing to a 2nd retrain. `BUILD_CHECKLIST.md:781`; `runs/lanes/calv1_seeddiag_20260708/diagnosis.json`
- Zero eval-label discipline violations across the entire CALV1 wave. `BUILD_CHECKLIST.md:743,747,758,776`

**Paddle**
- Fused 6-DOF estimator: IoU 0.11→0.26 / 0.03→0.36, jitter 23-53→2-5 deg/f, 0 undeclared teleports after 3 verify legs. `runs/lanes/racket_6dof_20260705/i1_fused_estimator/acceptance_record_v2.json`
- Ablation identified real driver (box evidence = whole gain) rather than assumed. `runs/lanes/racket_6dof_20260705/STATUS.md`

**Speed pipeline**
- 2141s→~532-565s (3.8x) with zero quality change. `runs/lanes/pipeline_speed_20260705/FINAL_REPORT.md`
- H100 validated end-to-end, 479.6s vs 1134.4s A100 ref (2.37x), version-stamp clean. `runs/lanes/w4_h100body_20260707/REPORT.md`
- Transport hardening folded 3 bonus catches (EMSGSIZE fallback, silent-degrade warning, overwrite guard). `runs/lanes/w5_transport_20260708/report.json`

**Product-infra-viewer**
- INFRA-0..5 full rebuild, all dark-flagged before enabling. `BUILD_CHECKLIST.md:716-724`
- Delete-cascade proven both directions on real S3/Mongo state. `BUILD_CHECKLIST.md:724`
- Owner critique #1 diagnosed read-only first (refuted denser-scheduling theory), then fixed with measured 4.1x/1.67x wins. `runs/lanes/w6_playbackdiag_20260708/playback_decision_table.json`

---

## 2. NEEDS REVISIT — ranked by leverage toward accurate+fast video→3D→feedback

**#1 — GATE-1b world round-trip decode fidelity FAIL, root cause untraced (body-world)**
- 262.35mm vs ≤1mm bar; mesh-skeleton divergence 53.50mm p95 vs ≤5mm bar. `runs/lanes/w6_close_errand_20260708/gate1b_raw_arm_report.json`
- Blocks lambda_foot, latent smoother, latent-interp playback, grounding_refine un-kill, AND product-infra's playback-quality ceiling simultaneously — a two-pillar blocker. `runs/manager/wave7_boot_prompt.md:29-36`
- Only clue: p95 identical raw-postchain vs default arm → decode-internal, post-chain-invariant, but no trace yet into world-frame transform / betas / translation-scale / persisted-fields gap.
- Severity: **high**. Suggested wave: **next (wave-7 #1 after ball, per boot prompt)** — dedicate a diagnosis-only lane (no retrain) tracing the mhr_decode path before any further latent work is attempted.

**#2 — Ball held-out corpus volume gap (1121 of ≥10-20k rows, ball-chain/data-engine)**
- `runs/lanes/w6_labelingest_20260708/report.json` ("5.6% to 11.2% of that bar"); gate defined at `NORTH_STAR_ROADMAP.md:176,315,874`
- Literal M1/M3 definition-of-done; every ball win to date (LoSO ranking, preprocessing fix, span-protection) is internal-val only, non-promotable. `BUILD_CHECKLIST.md:780`
- Compounded by an unmitigated sampling bias: current ranking rests on a disagreement-selected (not uniform) corpus. `BUILD_CHECKLIST.md` RULING A ("disagreement-corpus caveat BINDING")
- Severity: **high**. Suggested wave: ongoing owner-labeling push (~42-84 owner-hrs to bar) + structural fix — add a small periodic uniform-random audit sample alongside the disagreement queue so future comparisons aren't invalidated.

**#3 — Paddle fused 6-DOF estimator built, accepted, NOT wired into default E2E (P3-1)**
- "Oldest BUILT-NOT-WIRED orphan, 4 waves." `BUILD_CHECKLIST.md:775,780`
- Zero new research/GPU/owner time needed — pure dispatch. Highest ROI-per-effort item on this entire list.
- Severity: **high** (per critic; was medium). Suggested wave: **immediate — single wiring lane, wave-7**, ahead of Magnus/further paddle factor work.

**#4 — Browser-verify blocked 2 consecutive waves by INFRA-3 sign-in gate (product-infra-viewer)**
- `BUILD_CHECKLIST.md:742` ("BROWSER VERIFY BLOCKED x4 by INFRA-3 signin gate"); `BUILD_CHECKLIST.md:780`
- This is the mechanism that caught the fail-open HUD bug historically (`runs/lanes/ball_viewer_failclosed_fix_20260705/`); losing it while body-world mesh-cap, ball, and court all land viewer-facing changes is a silent, compounding regression risk.
- Severity: **high**. Suggested wave: **wave-7, small lane** — a dev-bypass arg for staging sign-in, not a redesign.

**#5 — P4-0 court-profile library, 0% built, deprioritized behind failing auto-find epic (court)**
- `NORTH_STAR_ROADMAP.md:1244-1252` ("0% built today"); owner override toward auto-find at `BUILD_CHECKLIST.md:747`
- Auto-find (P4-2/P4-3) has missed its ≤200px bar on every attempt (best 244.3px Burlington / 212.6px Wolverine, 0/8 and 2/8 pool containment, `BUILD_CHECKLIST.md:776`) and just committed a 2nd GPU retrain with no proof of fix (`pickleball-calv1unet2-a100-spot`, CREATING, `runs/manager/gpu_fleet.md`).
- Severity: **high** (per critic; was medium) — near-certain, cheap win (owner's own ≤3 courts) sitting idle while budget chases a repeatedly-missing epic.
- Suggested wave: **wave-7 — resequence P4-0 ahead of a 3rd auto-find retrain attempt**, or explicitly re-confirm owner override with the 244.3/212.6px evidence in hand.

**#6 — P5-1 speed gate uncertified against its own thresholds; the one attempted lever regressed (speed-pipeline)**
- Gate: Wolverine ≤400s, Outdoor ≤2x, six-run variance, bit-identical foot-slide. `NORTH_STAR_ROADMAP.md:1436-1444`. Closest evidence (H100 479.6s) sits above the 400s bar and is a caveated (not clean-room) comparison. `runs/lanes/w4_h100body_20260707/REPORT.md`
- S4 chunked-handoff lever regressed (1057.4s→1300.7s) and was reverted; no replacement implemented yet. `NORTH_STAR_ROADMAP.md:1436-1444`
- Severity: **high** (per critic; was medium) — the headline "3.8x" claim in `NORTH_STAR_ROADMAP.md:1427-1432` is repeated as settled fact while its own gate is unscored.
- Suggested wave: run one clean-room P5-1 gate-scoring lane before further speed claims are cited in a NORTH_STAR refresh.

**#7 — Owner 4-marker paddle GT session queued since wave-5, never booked (paddle-racket)**
- Only path to RKT VERIFIED. `NORTH_STAR_ROADMAP.md:1969`; re-surfaced every wave close, `BUILD_CHECKLIST.md:780`
- Severity: high (hard ceiling on promotion) but owner-gated, not engineering-gated. Suggested wave: bundle into a single owner-ask batch (see cross-pillar risk below) rather than re-surfacing individually each wave.

**#8 — seed_official's 486-row fine-tune underperformed control on LoSO-mean (0.6404 vs 0.6858), unresolved (ball-chain)**
- Booked "small-N noise suspect," never re-run. `BUILD_CHECKLIST.md:740`
- Severity: **medium** (per critic; was low) — seed_official is now the pillar's declared winner and presumptive base for further fine-tuning; an unresolved regression in its own lineage should be checked before more owner-labeling hours pour into extending the same recipe.
- Suggested wave: cheap re-run lane, wave-7, before next seed-based fine-tune round.

**#9 — GPU spot $/hr ambiguity (7x band, $0.57-$4.25/hr) blocking $/clip cost claims (speed-pipeline)**
- `runs/lanes/w4_h100body_20260707/REPORT.md` HONEST ISSUES #6; feeds P5-4/P7-3 pricing.
- Severity: medium. Suggested wave: owner-ask (real GCP invoice), not engineering — batch with other owner asks.

**#10 — Mesh byte-budget 300 vs 400 MiB + human_review-tier display, pending owner ruling (body-world/product-infra)**
- `BUILD_CHECKLIST.md` RULING C; `configs/racketsport/best_stack.json` (mesh.byte_budget_mib rev 2)
- Severity: medium. Suggested wave: single owner decision, bundle with #7/#9.

---

## 3. CROSS-PILLAR RISKS

1. **No unified GPU-fleet budget view.** Three high-severity blockers (GATE-1b root-cause, ball label re-score, court_unet_v2 retrain) each independently propose to consume fleet hours with no shared allocation decision. Risk: starving GATE-1b (blocks all downstream body/mesh work) in favor of a lower-yield retrain that has already regressed once. Needs a single owner-visible fleet-spend-vs-ask table before wave-7 dispatch.
2. **No unified owner-time/bandwidth queue.** Paddle 4-marker GT session (`NORTH_STAR_ROADMAP.md:1969`), 42-84 owner-labeling hours (`runs/lanes/w6_labelingest_20260708/report.json`), phone tests/game recording (`BUILD_CHECKLIST.md:780`), Roboflow API key re-issue (`NORTH_STAR_ROADMAP.md:247`) are all tracked per-pillar with no ranked consolidated backlog. If owner bandwidth — not GPU or engineering — is the true bottleneck, this queue needs explicit sequencing.
3. **No security/secrets/PII review** for the now-live product-infra build (JWT auth, Stripe scaffold, S3-hosted player video uploads). `BUILD_CHECKLIST.md:716-724` shipped dark-flagged but no secret-rotation, PII/retention, or auth-security pass exists in any pillar map before flags flip live.
4. **No input/capture-quality guardrail pillar.** P5-5b (ffprobe/orientation pre-flight) is unbuilt with orientation hardcoded "landscape" in 2 places (`NORTH_STAR_ROADMAP.md:1466-1470`). No pillar treats garbage-in protection for arbitrary end-user uploads as first-class — directly undermines the "trust > novelty" product philosophy once real users upload video.
5. **No licensing/compliance check on training-data supply chain** — Roboflow's 61,260-sample corpus (data-engine) and PnLCalib GPL checkpoints (court) are load-bearing inputs with no license-vs-commercial-product review, despite a Stripe monetization scaffold now existing (`BUILD_CHECKLIST.md:716-724`).

---

## 4. DOCS-STATE — stale/unclear, needs truth-up

| Doc | Problem | Evidence |
|---|---|---|
| `CAPABILITIES.md` ball row | Still shows stage1_official as current picture; doesn't reflect wave-6 finding that seed_official is the real LoSO winner (stage1_official fell below control at 0.2971). | pillar map note |
| `CAPABILITIES.md` line 100 (body row) | Predates GATE-1b legitimate-FAIL numbers and meshcap RULING C win; still cites older w4_freshproof/P2-2-unwired framing. | body-world pillar map |
| `NORTH_STAR_ROADMAP.md` P0-4 status (~699-717) | States corpus "~486 rows" — actual is 1121 as of `runs/lanes/w6_labelingest_20260708/report.json`. | data-engine pillar map |
| `NORTH_STAR_ROADMAP.md` P4-1 item (~1254-1261) | Still phrased as blocked ("patch does NOT apply") from a 2026-07-06 harsh-review note; `BUILD_CHECKLIST.md:753` confirms Wave-A actually landed on main 2026-07-08. Checkbox unchecked. | court pillar map |
| `NORTH_STAR_ROADMAP.md` P4-2/P4-3 items (~1266-1280) | Still reads as pure planning ("currently 213.3px") without the 2026-07-08 training run, GEO r3 diagnosis, or fused honest-miss numbers (244.3/212.6px). | court pillar map |
| `NORTH_STAR_ROADMAP.md` PHASE 5 header (1427-1432) | States "2141→532-565s" and "floor at 6-8min/clip" as settled while the S4 lever behind that floor regressed and was reverted; reads more settled than the underlying lane evidence supports. | speed-pipeline pillar map |
| `NORTH_STAR_ROADMAP.md` PHASE 7 ("110 Swift files") | Undercounts; actual is 140 (86 non-test + 54 test) as of 2026-07-08 per `find ios -name '*.swift' \| wc -l`. | product-infra-viewer pillar map |
| `TECH_BLUEPRINTS.md` PADDLE blueprint (PART C) | States "BUILT-NOT-WIRED until P3-1" — still accurate but should be checked against any post-wave-6 resequencing since P3-1 has now waited 4 waves past its "sequenced next" note (`BUILD_CHECKLIST.md:775`). | paddle-racket pillar map |