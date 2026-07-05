# PIPELINE STATUS — living document

Owner-requested standing document (2026-07-05): states what we are doing, what passes, and **highlights every failure case honestly**. Updated by the manager session as work lands. VERIFIED=0 everywhere — nothing below is promoted.

---

## 1. What we are doing right now

**ACTIVE PRIORITY (owner directive 2026-07-05): PIPELINE SPEED.** Make `scripts/racketsport/process_video.py` end-to-end as fast as possible. Placement/identity/calibration work is CLOSED (see §2) — 3 of 4 clips pass every computable gate, IMG_1605's one attributed FAIL is accepted as-is by the owner.

Speed baseline (final placement-lane runs, 2026-07-04, **with** no-force stage reuse — cold runs are slower):

| clip | frames (approx) | wall total | run dir |
|---|---|---|---|
| Wolverine | ~300 | **2141 s (36 min)** | runs/vp1_wolverine_20260704T215424Z |
| Burlington | ~600 | **1756 s (29 min)** | runs/vp1_burlington_20260704T215924Z |
| Outdoor | ~3796 | **3163 s (53 min)** | runs/vp1_outdoor_20260704T215924Z |
| IMG_1605 | ~813+ | **1521 s (25 min)** | runs/vp1_img1605_20260704T215924Z |

**Measured (2026-07-05):** the remote BODY stage is 96.9–98.4% of total wall on every clip; GPU inference itself is only ~2–5% of that. The real costs: (a) **GPU lock wait** — up to ~800s/clip when BALL training holds the A100 (scheduling contention, not code; ~0 on a quiet GPU); (b) **remote command overhead** — model load/warmup + serializing ~1.9–2.0GB of pretty-printed JSON that almost nothing downstream consumes; (c) **download** ~125s/clip for that payload. Evidence: `runs/lanes/pipeline_speed_20260705/TIMING_REPORT.md`.

**Results so far (landed + live-verified 2026-07-05, Wolverine, lock-free GPU, zero quality regressions on every run):** E2E **2141s → 1399s (S1) → 1285.6s (S2) → 1057.4s (S3)**, gates green and metrics bit-identical on four consecutive verification runs. S1: compact serialization, heavy monoliths opt-in download (1938MB→54MB), VM-built mesh index (viewer-consumable mesh for the first time). S2: batched rsync (transfers ~120s→~25s) + interior attribution. S3: in-memory index build (260→22.9s) + subprocess-level timing that produced the decisive fact: **actual GPU inference is 18.0s (15.3ms/person)** — the remaining ~810s is data plumbing (pickle handoff 376s + monolith payload assembly 343s + serialization 35s) for two ~430MB archival JSONs that the default flow never downloads. Round 4 (S4, in flight): binary array handoff + assemble/write monoliths only when `--fetch-body-monoliths`; projected E2E ~7–9 min. No cold-start effect exists (proven); warm-worker daemon dropped (would save only ~65s). Speed lane home: `runs/lanes/pipeline_speed_20260705/`, full trail in its `STATUS.md`.

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

1. **IMG_1605 foot slide gate FAIL: 0.330 m vs 0.030 m bar (ACCEPTED by owner 2026-07-05, not fixed).** Cause: the zero-distortion 15-point calibration cannot model the lens at the extreme left image edge (player-2 bbox x̄ ≈ 53 px). Stance phases there demand > 0.30 m pin corrections; they are dropped fail-closed (audit: `cap_exceeded_skips`, kind=phase_skipped in the foot_pin audit), so the real unpinned slide shows in the gate instead of being masked. 48 other phases pin cleanly (max 0.179 m). Durable fix if ever reopened: distortion-aware calibration for owner captures.
2. **IMG_1605 players 3/4 are rendered as EXCLUDED, not tracked-and-placed.** They are adjacent-court people behind the fence (proven: p4 median y = 8.32 m ≈ 1.6 m beyond the baseline; p3 pinned to a 0.2 m patch at the far corner its whole life). The world honestly renders 2 players. If a real 4-player owner capture arrives, membership verdicts must be re-checked.
3. **Per-player foot-slide p95 and Wolverine right-foot-to-centerline endpoint residual are NOT_COMPUTABLE** in the strict rollups (needs a skeleton3d parse harness — deferred). The pooled/max slide numbers above are real; the per-player breakdown is not measured.
4. **Mesh layer is `mesh_status=monolithic_unverified` on all four clips.** The 950 MB→30 MB mesh index splitter is landed and proven on 2 real meshes, but per-clip index builds were deferred (owner speed directive). Skeleton rendering is unaffected.
5. **Pickleball line-family / centerline evidence is advisory only** — `auto_centerline_evidence_ready=false` on all clips. Court calibration still rests on the 15-point manual/metric15 homographies.
6. **VERIFIED=0.** No documented repo acceptance gate (TRK IDF1, BALL M1, BODY world-MPJPE) has been run to promotion standard on these runs; trust bands remain fail-closed.
7. **BALL accuracy wall (other session's lane):** held-out shot missed honestly (0.6969 vs 0.7248 bar, 2026-07-04). Zero-shot public-data approaches are exhausted; next unlock is owner in-domain data. Not touched by this session.
8. **Standing operational gotchas:** remote BODY fails exit 1 (pydantic extra_forbidden) if new tracks/placement fields ship without re-syncing `schemas/__init__.py` to the VM; Outdoor-length clips need `--remote-command-timeout-s 7200`.

## 4. Where things live

- Placement lane (closed): `runs/lanes/joint_visual_placement_20260704/FINAL_REPORT.md`
- Speed lane (active): `runs/lanes/pipeline_speed_20260705/STATUS.md`
- Final run dirs: see table in §1; each contains `strict_rollup.{json,md}`, `membership.json`, `PIPELINE_SUMMARY.json`.
- Viewer: `cd web/replay && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort`, then `http://localhost:5173/?manifest=/@fs/<run_dir>/<clip>/replay_viewer_manifest.json`.
- Coordination: `BUILD_CHECKLIST.md` (channel), `CAPABILITIES.md` (canonical on conflicts).

*Last updated: 2026-07-05 by manager session (placement lane closed; speed lane opened).*
