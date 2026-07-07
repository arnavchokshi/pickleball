# PIPELINE STATUS — living document

Owner-requested standing document (2026-07-05): states what we are doing, what passes, and **highlights every failure case honestly**. Updated by the manager session as work lands. VERIFIED=0 everywhere — nothing below is promoted.

---

## 1. What we are doing right now

**WAVE-4 DOC RECONCILIATION (2026-07-07):** `VERIFIED=0` remains binding. Landed, evidence-complete updates: camera-motion decode-orientation policy + fail-safe mismatch semantics (cd0b59390; `runs/lanes/w4_cammotion_fix_20260707/report_r2.json`) and first-class `camera_motion_auto` summary keys (1588b110f; `runs/lanes/w4_integration_20260707/report.json`); stage-2 BALL sparse-review trainer/SST manifest/disagreement tooling (5b268aa6d; `runs/lanes/w4_ballcode_20260707/report.json`); harvest court calibration measured at 1/6 source manual_bar and 8/40 clips covered, teacher still deferred (83e090168; `runs/lanes/w4_court_harvestcal_20260707/report.json`); explicit remote-host requirement and committed-blob version stamps (dcc4dae42; 190dea09f; `runs/lanes/w4_fleethosts_20260707/report.json`; `runs/lanes/w4_syncstamp_20260707/report.json`); mesh-index warning telemetry distinguishes embedded-world absence from true mesh absence (684d03380; `runs/lanes/w4_burlmesh_fix_20260707/report.json`). Proof-dependent closeout numbers remain [PENDING wave-4 decisive proof — manager fills at closeout].

Speed baseline (final placement-lane runs, 2026-07-04, **with** no-force stage reuse — cold runs are slower):

| clip | frames (approx) | wall total | run dir |
|---|---|---|---|
| Wolverine | ~300 | **2141 s (36 min)** | runs/vp1_wolverine_20260704T215424Z |
| Burlington | ~600 | **1756 s (29 min)** | runs/vp1_burlington_20260704T215924Z |
| Outdoor | ~3796 | **3163 s (53 min)** | runs/vp1_outdoor_20260704T215924Z |
| IMG_1605 | ~813+ | **1521 s (25 min)** | runs/vp1_img1605_20260704T215924Z |

**Measured (2026-07-05):** the remote BODY stage is 96.9–98.4% of total wall on every clip; GPU inference itself is only ~2–5% of that. The real costs: (a) **GPU lock wait** — up to ~800s/clip when BALL training holds the A100 (scheduling contention, not code; ~0 on a quiet GPU); (b) **remote command overhead** — model load/warmup + serializing ~1.9–2.0GB of pretty-printed JSON that almost nothing downstream consumes; (c) **download** ~125s/clip for that payload. Evidence: `runs/lanes/pipeline_speed_20260705/TIMING_REPORT.md`.

**FINAL RESULT (lane closed 2026-07-05):** Wolverine E2E **2141s → 1144s (1.87x)** on a lock-free GPU with zero quality change (gates green, foot-slide metrics bit-identical across six verification runs). Transfers per clip **1.96GB → 76MB**; VM disk per dispatch ~2GB → <1GB; **mesh layer viewer-consumable for the first time** (VM-built 30MB windowed index); full per-phase instrumentation local+remote. Landed: compact/slim BODY artifacts (monoliths only with `--fetch-body-monoliths`), batched rsync, in-memory mesh-index build (260→23s), subprocess timing, robustness + contract fixes. Honest misses booked in `runs/lanes/pipeline_speed_20260705/FINAL_REPORT.md`: S4's chunk-streaming costs ~90–120s vs the S3 profile (run #4 hit 1057s — the single best run); handoff ~489s, assembly ~335s, warmup+load ~67s remain; measured floor if booked fixes land ≈ 6–8 min/clip. GPU lock contention with BALL training (~400–800s/clip) is scheduling, not code. Deep-dive evidence: lane `STATUS.md` + `FINAL_REPORT.md`.

## 2. Placement/identity/calibration outcome (lane CLOSED 2026-07-05 by owner)

Lane: `runs/lanes/joint_visual_placement_20260704/` — `FINAL_REPORT.md` (deliverable), `WOLVERINE_CHECKPOINT.md` (gate table), `STATUS.md` (full trail).

| clip | body gate | root jumps | foot slide (30 mm bar) | sides | rollup |
|---|---|---|---|---|---|
| Wolverine | PASS | 0 | PASS 18.4 mm | 1.0 ×4 | 0 FAIL, 2 N/C |
| Burlington | PASS | 0 | PASS 8.3 mm | 1.0 ×4 | 0 FAIL, 2 N/C |
| Outdoor | PASS | 0 | PASS 23.2 mm | 1.0 ×4 | 0 FAIL, 2 N/C |
| IMG_1605 | PASS | 0 | **FAIL 330 mm — accepted, attributed** | 1.0 ×4 | 1 FAIL, 1 N/C |

All four human-reported defects fixed and re-proven live: Wolverine two-per-side start; Burlington far-pair quadrant separation (double-undistortion removed); Outdoor identity swap (sidecar ID vote-mapping) + teleport (segmented smoothing, 1.64 m → 0.133 m); IMG_1605 renders only its two real players (membership exclusion) with camera-motion-compensated placement.

## 3. ⚠️ FAILURE CASES — read this section first

1. **IMG_1605 2026-07-05 foot-slide failure is superseded by later scoped BODY evidence, not by VERIFIED promotion.** Wave-3 closed the frozen 30mm max gate on fresh GPU internal-val clips, while wave-4 foot attribution measured the proposed skeleton-direct un-kill as negative: Burlington/Outdoor/IMG1605 accepted phases predict 34.6/33.6/48.4mm vs the frozen 30mm bar, so raw skeleton noise is still the binding constraint and grounding_refine stays an honest no-op (75e438223; `runs/lanes/w4_footattr_fix_20260707/report_r2.json`). Decisive wave-4 in-pipeline proof remains [PENDING wave-4 decisive proof — manager fills at closeout].
2. **IMG_1605 players 3/4 are rendered as EXCLUDED, not tracked-and-placed.** They are adjacent-court people behind the fence (proven: p4 median y = 8.32 m ≈ 1.6 m beyond the baseline; p3 pinned to a 0.2 m patch at the far corner its whole life). The world honestly renders 2 players. If a real 4-player owner capture arrives, membership verdicts must be re-checked.
3. **Per-player foot-slide p95 and Wolverine right-foot-to-centerline endpoint residual are NOT_COMPUTABLE** in the strict rollups (needs a skeleton3d parse harness — deferred). The pooled/max slide numbers above are real; the per-player breakdown is not measured.
4. **RESOLVED telemetry wording 2026-07-07:** mesh index is now built by default (VM in-memory, 23s) and viewer-consumable on previews, and viewer warnings now distinguish `missing_embedded_mesh_vertices` from true `missing_mesh_vertices` absence (684d03380; `runs/lanes/w4_burlmesh_fix_20260707/report.json`). Remaining mesh caveat: contact-dense (`ball_aware`) scheduling cannot use physical-contact triggers on a fresh default run because the default chain does not emit `events_selected.json` (wiring gap booked in `runs/lanes/wiring_audit_20260705/WIRING_TRUTH_TABLE.md`).
5. **Pickleball line-family / centerline evidence is advisory only** — `auto_centerline_evidence_ready=false` on all clips. Court calibration still rests on the 15-point manual/metric15 homographies.
6. **VERIFIED=0.** No documented repo acceptance gate (TRK IDF1, BALL M1, BODY world-MPJPE) has been run to promotion standard on these runs; trust bands remain fail-closed.
7. **BALL accuracy wall (other session's lane):** held-out shot missed honestly (0.6969 vs 0.7248 bar, 2026-07-04). Zero-shot public-data approaches are exhausted; next unlock is owner in-domain data. Not touched by this session.
8. **Array-native BODY path (payload collapse) is NOT accuracy-neutral yet:** stance lower-body protection not threaded (828→0 protected frames vs legacy, same worldhmr). OPT-IN OFF until fixed; fix recipe in `runs/lanes/payload_collapse_isolation_20260705/REPORT.md`.
9. **BALL P1-4a BVP remains PARTIAL:** real LOO per-holdout refit now exists and verifier confirmed 5 unique param sets, but D.3(b) protected-span preservation is not achieved; Magnus stays gated behind STEP 1, and the wave-5 design is frozen-baseline arc params as protected-span priors plus junction repair before validity gates (5633c4b48; `runs/lanes/w4_bvp_20260707/report_r3.json`; `runs/lanes/w4_bvp_verify_20260707/report.json`).
10. **Standing operational gotchas:** remote BODY now fails loud when `--remote-host` is omitted and points to `runs/manager/gpu_fleet.md`; `DEFAULT_REMOTE_HOST` is removed from code (dcc4dae42; `runs/lanes/w4_fleethosts_20260707/report.json`). Outdoor-length clips still need `--remote-command-timeout-s 7200`.

## 4. Where things live

- Placement lane (closed): `runs/lanes/joint_visual_placement_20260704/FINAL_REPORT.md`
- Speed lane (active): `runs/lanes/pipeline_speed_20260705/STATUS.md`
- Final run dirs: see table in §1; each contains `strict_rollup.{json,md}`, `membership.json`, `PIPELINE_SUMMARY.json`.
- Viewer: `cd web/replay && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort`, then `http://localhost:5173/?manifest=/@fs/<run_dir>/<clip>/replay_viewer_manifest.json`.
- Coordination: `BUILD_CHECKLIST.md` (channel), `CAPABILITIES.md` (canonical on conflicts).

*Last updated: 2026-07-07 (W4-F docs reconciliation; evidence-complete items only; VERIFIED=0).*
