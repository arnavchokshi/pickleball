# Joint Detection & Player Placement — Engineering Handoff

**Last updated:** 2026-07-03
**Author:** Fable (AI manager) → handing off to a new engineer
**Repo:** `/Users/arnavchokshi/Desktop/pickleball` · GitHub `arnavchokshi/pickleball` · main @ `83ef713d`
**Primary working video (NEW, as of this handoff):** `eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/source.mp4`

> Read this top to bottom before touching code. Sections 1–2 are the goal and the mental model. Section 6 (**Failure Cases**) is the most important part — it is where the remaining work is. Section 8 tells you exactly where everything lives.

---

## 1. The Final Goal (state this precisely)

Take an ordinary monocular video of a pickleball rally (single fixed camera, 4 players) and reconstruct an **accurate, watchable 3D world** you can scrub in a browser. Two things must be true at once, and they are in tension:

1. **Correct placement** — every player stands at the *exact* court location they occupy in the video. If a player's foot is on the kitchen line in the frame, their avatar's foot is on the kitchen line in the 3D world. This must hold for **far-from-camera players**, who are the hardest case.
2. **Accurate joints, fast** — each player's body pose (the skeleton, and a full mesh at key moments) is as anatomically correct as possible, with **minimal jitter**, while the whole pipeline runs at a **fast wall-clock time** (the tier decision below trades mesh cost for speed deliberately).

Everything in this repo is in service of those two sentences. The ball physics, shot taxonomy, and UI are downstream and comparatively solved; **joints + placement are the frontier.**

---

## 2. Mental Model — How a video becomes a 3D world

The pipeline is a sequence of stages in `scripts/racketsport/process_video.py` driven by `threed/racketsport/orchestrator.py`. Canonical stage order lives in `threed/racketsport/pipeline_contracts.py`. For player reconstruction the chain is:

```
video
 → calibration    (court homography + camera K,R,t from tapped/reviewed corners)
 → tracking       (YOLO26m + BoT-SORT-ReID → per-player bboxes)
 → placement      (NEW: where on the court each player stands)   ← Section 5
 → body           (SAM-3D-Body on GPU → 70-joint skeleton + mesh) ← Section 4
 → physics / ball_events / shots
 → world build    (virtual_world.json → confidence_gated_world.json)
 → replay viewer  (web/replay, localhost:5173)
```

Two artifacts carry the player through the world:
- **`skeleton3d.json`** — 70 joints per player per frame (the "stick figure").
- **`body_mesh.json`** (chunked into `body_mesh_chunks/`) — full mesh vertices, only at ~184 "ball-aware" frames (see tier decision).

The **viewer** (`web/replay/src/App.tsx`) renders skeletons as bones, meshes as solid bodies, plus paddles (wrist-proxy), the ball, and the court.

---

## 3. The Joint-Detection Journey (what we built and why)

### 3.1 The pivot: RTMW → SAM-3D-Body only
We started on **RTMW3D** (fast: ~44 ms/person) but its joints were metrically unstable — bone lengths swung frame to frame (CV 38.8%), limbs changed length, far players looked "terrible." The owner directive was unambiguous: **rip out RTMW, use SAM-3D-Body for everything.** SAM-3D-Body (`third_party/Fast-SAM-3D-Body/`) is a promptable human-mesh-recovery model producing a 70-joint MHR skeleton **and** a full mesh from a single image crop.

Measured trade (A100, `runs/sam3d_bodymode_bench_20260703T0211Z/`):
| Metric | RTMW3D (removed) | SAM-3D body-mode (production) |
|---|---|---|
| Speed batched | 32.2 ms/person | **32.2 ms/person** (after optimization) |
| Bone-length CV | 38.8% / 63.1% p90 | **2.5% / 5.7%** (15–20× steadier) |
| Bone error vs canonical | 44.2% median | **1.0%** (44× closer) |
| Elbow jitter p90 | 388 mm | **65 mm** (6× calmer) |

SAM-3D body-mode ended up **as fast as RTMW after optimization** and dramatically more accurate. `PoseStageRunner` is now a tombstone (refuses to run, points at BODY); `body` depends directly on `tracking`.

### 3.2 Speed: how we got SAM-3D to 32 ms/person
Real GPU work is in `scripts/racketsport/run_sam3dbody_batch.py`. Key wins, all A100-verified (`runs/a100_sam3d_validation2_20260703T0647Z/`):
- **`inference_type="body"`** — skips the hand-crop decoder (we don't need fingers), keeps the 70-joint skeleton + mesh head.
- **Real bucketed batching** — all 4 player crops + frames batched, static per-clip camera intrinsics, `torch.compile`.
- **`torch.inference_mode()`** wraps warmup + inference (a missing grad-guard here caused an OOM crash — see Failure 6.5).
- **Double warmup per bucket shape** (`compile_warmup_passes=2`) — killed a 12.8s first-call stall (cudagraph captures on the *second* call, so warm each shape twice). First call now 0.56s.
- Flag-gated knobs exist for further tuning (`crop_bucket_sizes`, `upstream_env` compile flags, `tier2_output_lite`) — defaults are measured-optimal on the 40GB A100.

**Gates PASS on production defaults: steady-state 32.23 ms/person (≤55 target), first-call 0.564s (≤1.0), ~$0.117 GPU/clip.**

### 3.3 The tier decision (speed vs mesh cost)
Full mesh everywhere is too slow. Policy `ball_aware_100` (`threed/racketsport/frame_rating.py`): render the expensive **mesh only near the ball / at contacts / during confident swings** (~184 frames on Wolverine); everywhere else emit the cheap **70-joint skeleton** only. This cut BODY-stage GPU time ~56%. **This is why meshes and skeletons don't both exist on every frame** — relevant to Failure 2.

### 3.4 The refine chain (this is where a lot of risk lives)
Raw SAM-3D joints go through `refine_sam3d_skeleton3d` in `threed/racketsport/pose_temporal.py`, in this order:
1. `_apply_motionbert_body17` — **no-op for SAM-3D** (runtime=None).
2. `_apply_one_euro` — temporal smoothing, per-joint-group params.
3. `_apply_bone_lengths` — enforce canonical bone lengths.
4. grounding — **disabled in production** (`worldhmr.py:369`).
5. `_apply_final_core_jitter_guard` — clamp implausible per-frame motion.
6. **wrist bone lock** (`apply_sam3d_wrist_bone_lock`) — project each wrist to canonical forearm length along the elbow→wrist direction (preserves swing timing exactly).
7. **contact splice** — overwrite wrists at contact frames from the mesh.

Then post-body: **foot pinning** (`threed/racketsport/foot_pin.py`) anchors planted feet to stance-phase positions, and (in staged worlds only) a **re-anchor** step shifts skeletons onto placement-corrected positions.

**Take this seriously:** the skeleton is the raw SAM-3D joints passed through *seven* transforms plus pinning plus re-anchoring. **The mesh is passed through none of them.** This is the root of Failure 2 (Section 6.2).

---

## 4. What "good" looks like now (the current best world)

Best staged Wolverine world: `runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/`
Viewer (start dev server first — Section 8.4):
`http://localhost:5173/?manifest=/@fs/Users/arnavchokshi/Desktop/pickleball/runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/replay_viewer_manifest.json`

Staging versions (each was a real experiment; keep for comparison):
| Ver | What it proved |
|---|---|
| v2 | foot pinning fixes sliding (slide p95 18.9mm) but wrists render +21% too long (raw smpl joints) |
| v3 | world-precedence fix → wrists exact (0.0%) but refine chain made **feet wander** (320mm) |
| v4 | **foot-wander root cause fixed** → feet 18.9mm AND wrists 0.0% together |
| v5 | **placement stage** → kitchen bias 0.46m→0.007m, far jitter 5–10× calmer, paddles in hands; but meshes displaced + slide regressed |
| **v5.1** | mesh re-anchor + stance-constant deltas → slide back to 18.9mm, meshes aligned, positions correct. **Current best.** |

v5.1 numbers (all measured, `runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/v4_v5_v51_comparison.json`): foot slide p95 18.9mm, kitchen-line bias 0.007m, paddle-to-wrist 0.14–0.16m, forearm error 0.0%.

---

## 5. The Placement System (OWNER-12, the "correct position" work)

**Diagnosis (measured, `runs/placement_diagnosis_20260703T1848Z/`):** the rendered player position was, to machine precision, the **detection bbox bottom-center unprojected through the court homography** — nothing else. Skeleton feet never influenced placement. That single fact caused both of the owner's original screenshots:
- Bbox bottom sits 10–14 px *below* the true feet → players pushed **+0.46m too deep at the kitchen line**.
- For crouched far players, the detector box **clips the feet off entirely** → **+1.1–1.8m behind the baseline**.
- Far-court: every pixel is 5–7cm of depth, so bbox noise → **0.77m median planted-foot wobble** (the "swimming").

**The fix (`threed/racketsport/placement.py`, new stage after tracking):** ground each player on their **foot keypoints** (native-2D ankle/heel/toe, and SAM-3D 2D foot keypoints via a new sidecar), not the bbox bottom; **undistort** pixels before projecting; weight each signal by pixel-noise pushed through the **homography Jacobian** (far court automatically gets honest large covariance); fuse and smooth with a **constant-velocity Kalman + RTS smoother** anchored by **stance phases** (planted feet = "we know exactly where they are"). It rewrites `tracks.json` world_xy in place (with backup + provenance), so downstream consumes better numbers with zero changes.

**Acceptance (all 7 targets PASS on Wolverine):** kitchen bias 0.46→**0.007m**; far planted wobble p90 1.92m→**~0**; far frame-to-frame speed 7.9→**0.73 m/s**; coverage unchanged; 0 court-bounds violations.

**Research verdict that shaped this** (`runs/manager/codex_lanes/reports/` + workflow `wf_fcb22b28-816`): the field-converged recipe is pose-keypoint ground points + distortion-corrected calibration + uncertainty-weighted temporal fusion with stance anchors. **World-grounded HMR translation (WHAM/TRAM/GVHMR) was evaluated and REJECTED** — 10–100× worse than plane projection at court scale. Use the skeleton for the *foot pixel*, never for *depth*.

---

## 6. MAIN FAILURE CASES (read this section twice)

These are the open problems. The two the owner flagged most recently (6.1, 6.2) are the priority.

### 6.1 Body lags behind the feet during fast movement  ⬅ PRIORITY
**Symptom (owner images 15/16):** when a player moves fast, the **feet stay correctly planted** but the **torso/upper body trails behind** — the body looks stretched or "catching up" to where the feet already are.

**What we know:**
- In the **staged worlds (v5/v5.1)** this is partly a *retrofit artifact*. Placement corrects the *anchor* trajectory, but the v5.1 skeletons were grounded on the GPU against the *old* anchors and then shifted post-hoc using **stance-constant deltas**: inside a planted stance the whole body uses one constant correction, but **between stances** the body follows the raw GPU motion + an interpolated delta rather than the smoothed placement trajectory. Measured divergence between body and its floor marker reaches **p90 1.86m for the far player (P3) between stances** (`runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/`, v5.1 report issue #2). In-stance is tight; the lag appears during the fast transitions between stances — exactly the owner's observation.
- There is likely **also a genuine version in the native pipeline**: the root/EMA smoothing (`worldhmr.py:780-848`, alpha 0.65 + speed clamp) and the one-euro joint smoothing add lag on fast direction changes.

**The durable fix (recommended):** stop re-anchoring skeletons *post-hoc* in staging. Instead **ground the skeleton on the placement anchors at refine time** — i.e., have the body stage / worldhmr consume the placement-corrected `track_world_xy` so the whole body is placed on the smoothed trajectory from the start, with no separate delta to interpolate. A **fresh end-to-end run** (Section 7) produces exactly this and should be the first thing you validate. If lag persists natively, tune the root smoother to be **stance-aware** (tight during stance, low-lag during transitions) rather than a uniform EMA.

### 6.2 Skeleton shape ≠ mesh shape (mesh is more accurate)  ⬅ PRIORITY + owner's sharp question
**Symptom (owner image 17):** at a mesh frame, the solid **mesh is an accurate lunge pose**, but the **stick skeleton is distorted** — limbs splayed, stretched, going the wrong way. The owner correctly asks: *they come from the same SAM-3D model, so why is the skeleton so much worse?*

**The answer (this is the key insight for the next engineer):** they start identical but **do not stay identical**.
- The **mesh** (`pred_vertices`) is **raw SAM-3D output** — zero post-processing.
- The **skeleton** (`pred_keypoints_3d`) is the raw joints run through the **entire refine chain** (Section 3.4): one-euro smoothing, bone-length enforcement, jitter guard, wrist bone-lock, contact splice, foot pinning, and staged re-anchoring. Each transform assumes "normal" motion; **extreme/fast lunge poses violate those assumptions and get distorted.** The mesh looks better because *nothing touched it.*
- Compounding it: on the staged worlds the **mesh vertices carry the original A100 arm/wrist geometry** (the mesh layer never received the wrist bone-lock — v5.1 report issue #4), while the skeleton did — so they were literally corrected differently.

**Fix direction (recommend evaluating, in order):**
1. **Derive the rendered skeleton from the mesh**, so they are consistent by construction. The MHR head builds keypoints from the skinned mesh vertices (`third_party/Fast-SAM-3D-Body/sam_3d_body/models/heads/mhr_head.py:589-616`) — a joint regressor on `pred_vertices` gives a skeleton that always matches the mesh silhouette.
2. Or **drastically reduce the refine chain** — the fact that the untouched mesh beats the heavily-processed skeleton is strong evidence we are **over-processing**. Measure raw-joints-vs-refined per stage on extreme-pose frames (the `sam3d_foot_wander` per-stage instrumentation pattern in `runs/sam3d_foot_wander_20260703T1024Z/` is the template — it already caught one-euro damaging feet). Apply corrections *only* where they measurably help.
3. Apply the same geometric corrections (wrist lock, foot pin) to the **mesh** too, so if both are kept they agree.

### 6.3 Far players are worse than near players (general)
Every failure above is amplified for far-from-camera players: 5–7cm depth per pixel, detector box clipping, keypoint confidence drops. Placement (Section 5) attacks this directly and got far jitter down 5–10×, but it is the permanent stress test. **Always evaluate far players specifically** — a fix that works near-camera may do nothing far.

### 6.4 Jitter is partly clamp-masked, not organically clean
The "root jitter p90 ≈ 0.100 m/frame" for 3 of 4 players is the **3.0 m/s core-body speed clamp ceiling** engaging on 16–32% of frames, not genuinely smooth motion. Independent hip-midpoint measurement shows body sway is largely unchanged v4↔v5.1. **Don't trust the jitter column as "body motion is smooth"** — it measures the (placement-smoothed) floor anchor. Real smoothness requires fixing 6.1/6.2 upstream.

### 6.5 Historical / infrastructure failures already fixed (so you don't re-hit them)
- **OOM crash**: bucketed inference lost `torch.no_grad()` → gradient graph through a 1.3B backbone → CUDA OOM. Fixed with `inference_mode()`. If you refactor the batch runner, keep it.
- **Compile stalls**: warm each bucket shape **twice** (cudagraph records on call 2).
- **Viewer default stick figures**: the world builder must emit top-level `joint_names` (70 MHR names); the viewer only auto-names 133-joint skeletons and silently drew placeholders otherwise. Fixed in `virtual_world.py`; keep it.
- **Silent skeleton bypass**: `virtual_world.py` used to prefer raw smpl joints over the corrected skeleton at every timestamp. Fixed (skeleton3d wins, smpl fills gaps). Regression test exists.
- **World `transl_world` still stale**: at smpl-covered timestamps the world's `transl_world` still carries OLD A100 anchors (v5.1 issue #3). Viewer renders joints/floor (both corrected) so no *visual* effect, but any consumer of `transl_world` misplaces bodies. Flagged for the world-builder.

### 6.6 Smaller known imperfections (bookkept, not blocking)
- Body "trust band" text is stale (describes an old dispatch run).
- Player-1 z-span borderline (~1.39m, scale-suspect) in every generation.
- `virtual_world.json` (intermediate) is not foot-pinned; only the rendered `confidence_gated_world.json` is.
- Paddles are a **wrist-proxy** (owner accepted as "honestly pretty good"); true 6-DoF paddle needs a marker capture session.

---

## 7. How to run it (and the FIRST thing the next engineer should do)

**First action: a fresh end-to-end run on the new primary video, then validate 6.1/6.2 natively.** The staged v2→v5.1 worlds are retrofits of a single old GPU dispatch; every remaining quirk (6.1, 6.2, 6.5 stale-transl) is partly a retrofit artifact. A clean run through the *current* pipeline places, grounds, locks, and pins everything natively — it is the real test of where we stand.

```bash
# from repo root, .venv active, GPU VM reachable (Section 8.3)
.venv/bin/python scripts/racketsport/process_video.py \
  --video eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/source.mp4 \
  --clip outdoor_webcam_iynbd_1500_long_high_baseline \
  --run-dir runs/e2e_outdoor_<DATE> \
  --force --rally-gating --ball-track
# (a fresh-Wolverine variant of this was queued but not completed — spend limit)
```

**⚠️ The new primary video (outdoor) has an unresolved court calibration.** It is a protected held-out eval clip; its auto-calibration found only 7/9 court lines and **fails the calibration gate** (`process_video.py:785`). Options for the next engineer / owner:
1. **Tap the court corners** for the outdoor clip (normal product flow, cleanest), or
2. Use the existing reviewed `court_calibration_metric15pt.json` in the clip's labels for **render-only** demo use (human-surveyed geometry, never model scoring). This is an **owner decision** — it was left open.
Placement quality is bounded by calibration quality, so **resolve this first** for the outdoor clip.

**Verify a staged world in the browser (headless):**
```bash
.venv/bin/python scripts/racketsport/verify_process_video_viewer.py \
  --manifest <run_dir>/replay_viewer_manifest.json --out-dir <run_dir>/viewer_verify
```

**Re-run the placement acceptance harness** (the numeric bar for "correct placement"):
`runs/placement_stage_20260703T1938Z/diagnose_placement_acceptance.py` (adapt paths).

---

## 8. Where everything lives

### 8.1 Core pipeline code
| Concern | File |
|---|---|
| Stage orchestration | `threed/racketsport/orchestrator.py` |
| CLI entrypoint | `scripts/racketsport/process_video.py` |
| Stage order/contracts | `threed/racketsport/pipeline_contracts.py` |
| **Placement (positions)** | `threed/racketsport/placement.py` + `scripts/racketsport/apply_placement.py` |
| SAM-3D GPU batch | `scripts/racketsport/run_sam3dbody_batch.py` |
| Skeleton refine chain | `threed/racketsport/pose_temporal.py` |
| Grounding / root anchor | `threed/racketsport/worldhmr.py` |
| Wrist bone lock | `apply_sam3d_wrist_bone_lock` in `pose_temporal.py` |
| Foot pinning | `threed/racketsport/foot_pin.py` + `scripts/racketsport/apply_foot_pin.py` |
| Mesh-tier policy | `threed/racketsport/frame_rating.py` |
| Foot-point (legacy bbox) | `threed/racketsport/person_fast.py:30-31` |
| World builder | `threed/racketsport/virtual_world.py` |
| Schemas | `threed/racketsport/schemas/__init__.py` |
| Viewer | `web/replay/src/App.tsx` (+ `viewerData.ts`) |
| Upstream model (read-only) | `third_party/Fast-SAM-3D-Body/` |

### 8.2 Key evidence / result dirs
- Placement diagnosis (root cause + failure forensics): `runs/placement_diagnosis_20260703T1848Z/`
- Placement stage acceptance: `runs/placement_stage_20260703T1938Z/`
- Foot-wander per-stage instrumentation (the debugging template): `runs/sam3d_foot_wander_20260703T1024Z/`
- Wrist lock: `runs/sam3d_wrist_bone_lock_20260703T0906Z/`
- A100 speed/accuracy validation: `runs/a100_sam3d_validation2_20260703T0647Z/`
- SAM-3D vs RTMW bench: `runs/sam3d_bodymode_bench_20260703T0211Z/`
- Best staged world: `runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/`
- E2E timing (old, full-mesh): `runs/e2e_timing_20260703T0818Z/`
- Research reports: `runs/manager/codex_lanes/reports/`
- Board/plan: `BUILD_CHECKLIST.md`, `MASTER_PLAN.md`

### 8.3 GPU
- VM: `arnavchokshi@34.126.67.233`, A100-SXM4-**40GB** (not 80GB), spot.
- SSH: `ssh -i ~/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=configs/ssh/a100_known_hosts arnavchokshi@34.126.67.233` (run from repo root).
- **Cold-start from scratch is proven (258s):** `scripts/racketsport/gpu_cold_start.sh` + `docs/racketsport/GPU_COLD_START.md`. A fresh VM needs `HF_TOKEN` (SAM-3D-Body weights are license-gated to the owner's HF account — the one non-scriptable step).
- **Keep the VM repo on latest main** before any GPU test: `git -C /home/arnavchokshi/pickleball_train_main pull`. A stale checkout silently ran old code and crashed a run (Failure history).
- VM disk runs tight — `gpu_cold_start.sh` and the cleanup notes in `BUILD_CHECKLIST` handle it.

### 8.4 Viewer / dev server
```bash
cd web/replay && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
```
Then open the manifest URL (Section 4). The dev server serves any local file via `/@fs/<absolute-path>`. `.venv` needs Python Playwright for the headless verifier (`pip install playwright && playwright install chromium`).

### 8.5 Reproducibility / discipline
- Everything is committed & pushed (main @ `83ef713d`). Model weights are **not** in git (manifest + hash fetch).
- Protected eval clips: **Outdoor + Indoor** — never read their `labels/` for scoring or training; running the pipeline for demo worlds is owner-authorized, labels are not.
- Every new `scripts/racketsport/*.py|*.sh` CLI needs a direct-CLI reference test (or `test_scaffold_tool_index.py` fails).
- Local test runner: `.venv/bin/python -m pytest tests/racketsport/... -q`. Torch-dependent tests use `pytest.importorskip("torch")` (torch is on the GPU box, not this Mac).

---

## 9. Recommended next steps (priority order)

1. **Resolve outdoor-clip calibration** (Section 7) — placement is bounded by it.
2. **Fresh end-to-end run** on the outdoor clip → validate whether 6.1 (body lag) and 6.2 (skeleton≠mesh) persist *natively* vs. being staging retrofit artifacts. Do not debug the staged v5.1 world for these; debug a fresh run.
3. **Fix 6.2 (skeleton≠mesh):** evaluate deriving the skeleton from the mesh, and/or measure the refine chain per-stage on extreme poses and cut what doesn't help. The untouched mesh beating the processed skeleton is the loudest signal in the whole system.
4. **Fix 6.1 (body lag):** make grounding consume placement anchors at refine time (kill the post-hoc re-anchor), and make root smoothing stance-aware.
5. **Then** the multi-video demo (Wolverine, Burlington, Outdoor, Indoor) and the remaining bookkept items (6.6).

---

*This document is the single source of truth for the joint-detection + placement effort. When you change the system, update Sections 4, 6, and 9.*
