# Sway Body — Build Checklist & Agent Coordination

**This is the operational source of truth for building Sway Body.** The lead build agent (Codex) and every subagent it spawns coordinate **here** — it is how everyone knows who is doing what, what is done, and what remains. Agents cannot talk to each other directly; **this file + git commit messages are the only communication channel.**

**Read order for any agent before doing anything:** (1) this file, (2) `SWAY_BODY_PICKLEBALL_MVP.md` (what/why), (3) `TECH_STACK.md` (which technology + why — **§2.3 Model Registry is the canonical exact model/variant/weights for every stage; never use a variant that contradicts it**), (4) the relevant phase in `IMPLEMENTATION_PHASES.md` (exact build + test steps), (5) `ACCURACY_AND_TRAINING.md` (datasets, training, validation gates). Each task below names its phase — go there for full detail; do not re-derive it.

**Model variants are benchmarked + human-approved, not assumed:** `TECH_STACK.md §2.3` lists each stage's default + candidates; **all H100/on-device speeds in this repo are estimates.** Task **EVAL-0** benchmarks the candidates on real clips, **renders side-by-side comparison videos, and requires human approval before locking the variant (§1.6) — unless one option is obviously better, in which case it may auto-finalize and log it.** The approved choice is locked in `models/MANIFEST.json` (live = lightest usable; offline = recompute with the accurate variant). Benchmark **Fast SAM-3D-Body first** (pacing item; sets the MIG geometry).

**Product scope reminder:** **single static camera** is the product focus. Multi-camera is FUTURE; multi-view is a TRAINING-TIME-ONLY technique that ships a single-camera model. Native **iOS Swift** client + GPU **server**.

---

## 1. Coordination protocol (mandatory)

### Status values
`TODO` → `CLAIMED` → `IN-PROGRESS` → `IN-REVIEW` → `DONE` → `VERIFIED`. Plus `BLOCKED` (with reason) and `PENDING-APPROVAL` (awaiting the human on a decision — see §1.6).

- **TODO** — not started, available to claim (if its dependencies are `VERIFIED`).
- **CLAIMED** — an agent has taken it; Owner set. No one else may touch it.
- **IN-PROGRESS** — actively being built.
- **BLOCKED** — cannot proceed; the `Notes` column states why and what's needed.
- **PENDING-APPROVAL** — a model/variant (or other non-obvious) decision is waiting on the human; comparison videos rendered and linked. Does NOT unblock downstream until approved (§1.6).
- **IN-REVIEW** — built; the task's own test passes; awaiting the lead's gate check.
- **DONE** — code complete, task self-test green.
- **VERIFIED** — the lead agent ran the phase **Acceptance Gate** (from `IMPLEMENTATION_PHASES.md`) and it passed. Only `VERIFIED` unblocks downstream tasks.

### Rules
1. **Claim before you build.** Edit the task row: set `Owner` + `Status=CLAIMED`, commit `claim <TASK-ID>`. One owner per task at a time.
2. **Respect dependencies.** Do not start a task until every task in its `Deps` is `VERIFIED`. If you need something not yet ready, pick another `TODO` whose deps are met.
3. **Own only your files.** Each task lists the files/modules it owns. Do not edit files owned by another `CLAIMED`/`IN-PROGRESS` task — coordinate via a `BLOCKED` note instead.
4. **Update status as you go.** `CLAIMED`→`IN-PROGRESS` when you start; `BLOCKED` with a reason if you stall; `IN-REVIEW` when your task's test passes.
5. **Finish with a handoff.** On completion: set `Status=DONE`, fill the **Handoff log** (§5) with: what you built, the artifacts/files produced, how you tested it, and what the next agent needs to know. Commit `done <TASK-ID>: <summary>`.
6. **Only the lead verifies.** The lead agent runs the phase Acceptance Gate on GPU test clips; pass → `VERIFIED`; fail → back to `IN-PROGRESS` with failing-gate notes.
7. **No phase advances on red.** A downstream phase's tasks stay `TODO` until their upstream gate tasks are `VERIFIED` (phase-gate discipline from `IMPLEMENTATION_PHASES.md §0.1`).
8. **Log decisions.** Any deviation, model swap, or assumption goes in the **Decisions log** (§6) so others see it.
9. **Keep commits scoped to one task** and prefix the message with the TASK-ID.

### Definition of Done (global)
A task is `VERIFIED` only when: code merged, its module unit tests pass, the phase Acceptance Gate (numeric, on real GPU test clips spanning the varied-camera matrix — run **through the §1.5 GPU lease/queue**) passes, artifacts conform to the JSON schemas in `IMPLEMENTATION_PHASES.md`, the Handoff log entry is written (incl. the recorded gate metric), **and any model/variant choice it depends on has cleared the §1.6 approval gate.**

### 1.6 Decision approval gate — human-in-the-loop on variant choices (mandatory)

**Whenever a task must choose between candidate models/variants (EVAL-0 and any per-stage variant pick), it does NOT silently finalize.** It must let the human compare and approve, *unless one option is obviously better.*

1. **Render comparison videos.** For each candidate, render its ACTUAL output overlaid on the **same ≥3 real test clips** spanning the varied-camera matrix (e.g., detection boxes / pose skeleton / SMPL mesh / ball track / foot-lock before-after / racket 6DoF / replay), as **side-by-side comparisons**, each labeled with the candidate name + its measured accuracy-gate metric + latency + VRAM. Save to `runs/eval0/<stage>/compare/` and write `runs/eval0/<stage>/variant_selection.md` (metrics table + links to every video).
2. **Auto-finalize ONLY if obvious.** The agent may lock a variant on its own — logging it in the **Decisions log (§6)** as `auto-finalized (obvious)` and still saving the videos — only when the winner is unambiguous: it **passes the gate AND is Pareto-dominant** (≥ as accurate AND ≥ as fast/cheaper than every alternative), OR every alternative **fails the gate** while it passes, OR it beats all others by a **large, clear margin** (e.g., well past the gate with no meaningful speed/VRAM penalty).
3. **Otherwise BLOCK on the human.** Set the task `Status = PENDING-APPROVAL`, note the link to `variant_selection.md` + the comparison videos in the `Notes` column and the Handoff log, and **stop** — do not lock the variant, do not start dependents that need the final pick. The human reviews the videos and chooses.
4. **On approval:** record `approved: <variant> by <human> on <date>` in the Decisions log, lock the choice in `models/MANIFEST.json`, set status forward (`IN-REVIEW`/`VERIFIED`).
5. **Scope:** this applies to every model-variant decision and to any *other* non-obvious decision with close alternatives (e.g., a calibration method, a smoothing filter) — surface comparison artifacts and request approval rather than silently picking. Obvious/no-alternative choices proceed normally.

---

## 1.5 GPU coordination — ONE shared H100 (mandatory)

**All agents share a single NVIDIA H100 80GB (GCP A3) for testing/training.** Naive parallel GPU use → OOM/crashes. The rule set below lets CPU work run fully parallel while GPU work is isolated and serialized.

### The #1 rule: CPU work never touches the GPU lease
Tasks tagged **[CPU]** (code, schemas, data download/convert, iOS Swift app, web viewer, CPU post-processing, MuJoCo-CPU, report formatting) run **immediately and in parallel** — they must never block on the GPU. Only **[GPU]** tasks (model training/fine-tune, heavy server inference, and **every eval/test run on test clips**) acquire the lease. (iOS on-device models run on the *phone's* Neural Engine, not this H100 — so IOS-* tasks are [CPU] w.r.t. this lease, except Core ML conversion/validation.)

### Static MIG geometry (don't flip per-job — drain+reset churn is too costly)
Pick ONE mode per session:
- **Eval/dev mode (default): `2 × 3g.40gb`** — the inference pipeline + parallel eval agents each get hardware-isolated 40 GB (a buggy agent OOMs only its own slice, not its neighbors). Use **MIG, not MPS** (autonomous agents need hard fault/memory isolation).
- **Training mode: `1 × 7g.80gb`** — one exclusive holder fine-tunes (SAM-3D-Body fine-tune needs ~40–55 GB → full GPU); eval agents pause.

### Lease/queue convention (local disk, `flock` — NOT NFS)
```
/run/gpu-lease/
  mode                 # "eval" | "training"
  full-gpu.lock        # exclusive; held by the one training/full-GPU job
  slots/slotN.lock     # one flock per MIG eval instance
  slots/slotN.uuid     # MIG-<uuid> for CUDA_VISIBLE_DEVICES
  heartbeat/<pid>      # pid+epoch ts; TTL watchdog evicts stale locks
```
- **[GPU] eval/inference task:** `flock -n` a free slot → `export CUDA_VISIBLE_DEVICES=$(cat slotN.uuid)` → run within that slice's VRAM budget → release. If none free, block on `flock` (FIFO queue) and meanwhile pick a [CPU] task.
- **[GPU] training task:** acquire exclusive `full-gpu.lock`; if it fits 40 GB just take a slot, else drain slots → switch `mode→training` → reconfigure `7g.80gb` → train → restore eval geometry on release.
- **Heartbeat + TTL** reclaims locks from dead agents. **Monitor with DCGM** (`dcgmi dmon -e 203,204,252`); stop DCGM before any `--gpu-reset`/mode switch (it counts as a GPU client).

Helper (`scripts/gpu-eval-run.sh`): grab a free MIG slot via `flock`, set `CUDA_VISIBLE_DEVICES`, run the command, release — see `IMPLEMENTATION_PHASES.md` Phase 0 for the script.

### VRAM budget (80 GB; estimates — measure SAM-3D-Body FIRST, it paces the pipeline)
| Job | VRAM | Fits |
|---|---|---|
| Full inference pipeline (all models resident) | ~18–28 GB | 3g.40gb slice |
| SAM-3D-Body **inference** (largest single model) | ~10–20 GB | 3g.40gb slice |
| SAM-3D-Body **fine-tune** (AdamW states) | ~40–55 GB | **needs 7g.80gb (training mode)** |
| YOLO26 / RTMW fine-tune | ~8–15 GB | a slice (eval-tier lease) |
| 3 parallel eval agents (YOLO+TrackNet+RTMW, no SAM) | ~20 GB each | 3× 1g.20gb |

### GPU-lease tasks (everything else is [CPU], no lease)
**[GPU]:** **EVAL-0 (variant benchmarking)** · DATA-2, DATA-3, DATA-4 (training/auto-label inference) · CAL-3 (train keypoint net) · TRK-1 (eval runs) · BODY-1, BODY-2, BODY-3, BODY-4 · FOOT-1 (eval) · BALL-1, BALL-2, BALL-4 (train/eval) · RKT-1 (train/eval) · MET-1 (eval) · SHOT-1 (train) · RPL-1 (render bake, if GPU-rendered) · **EVAL-1, EVAL-2, EVAL-4 (all test-clip runs)**. FOOT-2 physics is [GPU] only if using MuJoCo-MJX/PHC training, else CPU.

### Testing is a GPU job
Every phase Acceptance Gate is run on test clips **through this lease/queue**. A task is `VERIFIED` only after its numeric gate passes on the shared H100 and the result is recorded in the Handoff log (§5).

---

## 2. Workstreams & dependency graph

```
ENV ──┬─> IOS (capture/calibration/fast-tier/viewer)        [iOS client track]
      └─> DATA (datasets, auto-label, fine-tune)             [runs alongside]
            |
ENV ─> CAL (court calib, seeded by ARKit sidecar)
            └─> TRK (detect/track/ID)
                  └─> BODY (Fast SAM-3D-Body + world-grounding)
                        ├─> FOOT (foot-skate kill + physics)
                        ├─> BALL (track + audio events + 3D physics)
                        │     └─> RKT (racket 6DoF; needs BALL for rebound-validate)
                        └─> MET (metrics + insights + confidence)  [needs FOOT, BALL, RKT]
                              ├─> SHOT (shot class + drill verify)
                              └─> RPT (LLM copy + report + viz)
                                    └─> RPL (3D replay: bake USDZ+GLB, RealityKit + web viewers)
EVAL  ── spans everything (harness + per-phase gates + regression CI)
```
Critical path: **ENV → CAL → TRK → BODY → FOOT/BALL → RKT → MET → RPT → RPL.** The **IOS** and **DATA** tracks run in parallel from ENV.

**Parallelization model (max throughput on one H100):** the lead spawns subagents to work concurrently. **[CPU] tasks run fully in parallel, anytime** (their deps permitting) — IOS app, web viewer, schemas, data download/convert, module scaffolding, CPU post-processing, report formatting. **[GPU] tasks** (training, heavy inference, all test-clip eval) **serialize through the §1.5 lease/queue** (or run concurrently in separate MIG eval slots when they fit). So: keep many CPU tasks in flight; let GPU tasks queue. FOOT and BALL can proceed in parallel once BODY is VERIFIED; SHOT/RPT can proceed once MET is VERIFIED.

---

## 3. Checklist

Columns: **☐** (done) · **ID** · **Task** · **Owns (files)** · **Deps** · **Phase** · **Owner** · **Status**. The **Phase** column maps each task to `IMPLEMENTATION_PHASES.md`: iOS client tasks → **Client Track C1–C3**; server tasks → the named **Pipeline Track phases (0–11)** — go there for the exact build steps + Acceptance Gate. (Phase names, not numbers, are used for server tasks so they stay valid if phases are renumbered.) **GPU vs CPU:** the **[GPU] tasks that must use the §1.5 lease are enumerated in §1.5**; every other task is **[CPU]** and runs in parallel without the lease.

### ENV — Environment & scaffolding
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☑ | ENV-1 | Server env, deps, repo scaffolding under `threed/racketsport/`, `models/MANIFEST.json` | repo skeleton, env files | — | Phase 0 | Codex | DONE |
| ☑ | ENV-2 | Fetch + checksum model checkpoints (incl. verify-before-commit flags + fallbacks) | `models/` | ENV-1 | Phase 0 | Codex | DONE |
| ☐ | ENV-3 | iOS Xcode project scaffolding (`ios/`), Swift package layout | `ios/` | — | Phase 0 | Codex | BLOCKED |
| ☐ | ENV-4 | NvDEC ingest + clip QC + capture-quality scoring (server) | `ingest.py` | ENV-1 | Phase 0 | Codex | IN-PROGRESS |

### IOS — iOS client (capture, calibration, fast tier, viewer)
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | IOS-1 | AVFoundation locked capture (exposure 1/500–1/1000 s, focus, WB, fps/format policy, landscape, HEVC/ProRes record, live frames) | `ios/Capture/` | ENV-3 | C1 | | TODO |
| ☐ | IOS-2 | ARKit setup/calibration pass → intrinsics + 6DoF pose + court plane → sidecar; manual tap fallback | `ios/Calibration/` | IOS-1 | C1 | | TODO |
| ☐ | IOS-3 | On-device fast tier (Apple Vision 2D/3D/hand/seg + YOLO Core ML ball/racket) → preview | `ios/FastTier/` | IOS-1 | C2 | | TODO |
| ☐ | IOS-4 | Capture-quality guidance UX (corner visibility, tracking state, exposure, blur risk, CoreMotion level/shake, iOS26 button capture) | `ios/Guidance/` | IOS-2, IOS-3 | C2 | | TODO |
| ☐ | IOS-5 | Upload pipeline (trimmed clip + sidecar + on-device pose prior + LiDAR depth if Tier A) | `ios/Upload/` | IOS-2 | C1 | | TODO |
| ☐ | IOS-6 | RealityKit/USDZ replay viewer (RealityView virtual camera, free-viewpoint gestures) | `ios/Replay/` | RPL-1 | C3 | | TODO |

### CAL — Court calibration (server)
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☑ | CAL-1 | Court templates (PB/tennis) + zones (NVZ, service boxes) + net plane from regulation geometry | `court_templates.py`, `court_zones.py`, `net_plane.py` | ENV-1 | Court Calibration | Codex | DONE |
| ☐ | CAL-2 | Per-clip calibration: seed from ARKit sidecar → solvePnP (6-DoF) + multi-frame averaging + reprojection gate; manual-tap fallback; capture-quality score | `court_calibration.py`, `intrinsics.py` | CAL-1, IOS-2 | Court Calibration | Codex | IN-PROGRESS |
| ☐ | CAL-3 | (Phase-2) Train auto court-keypoint net (fine-tune TennisCourtDetector + synthetic viewpoints) | `court_keypoint_net.py` | CAL-2, DATA-1 | Court Calibration | | TODO |

### TRK — Person detection, tracking, doubles ID
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | TRK-1 | YOLO26m + BoT-SORT-ReID detect/track; court-polygon filter; ground-plane association; N-lock + coach 1-tap anchor | `person_fast.py`, `track_lock.py`, `doubles_id.py` | CAL-2 | Person Detection/Tracking | Codex | IN-PROGRESS |

### BODY — 3D body mesh (core)
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | BODY-1 | Deep tier: Fast SAM-3D-Body per player crop (MHR→SMPL via MLP) | `hmr_deep.py` | TRK-1 | 3D Body | | TODO |
| ☐ | BODY-2 | Our world-grounding: project per-frame to world via known camera + court Z=0 → temporal smooth | `worldhmr.py` | BODY-1, CAL-2 | 3D Body | | TODO |
| ☐ | BODY-3 | Fast tier: camera-space mesh (SAT-HMR / Multi-HMR 2) for preview | `hmr_fast.py` | TRK-1 | 3D Body | | TODO |
| ☐ | BODY-4 | Racket-motion fine-tune (BEDLAM2→AthletePose3D→CalTennis→RICH→AMASS) + world-MPJPE eval | `scripts/finetune_pose.py` | BODY-1, DATA-2 | 3D Body | | TODO |

### FOOT — Foot-skate elimination & physics
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | FOOT-1 | Foot-contact detection vs known Z=0 + zero-velocity + CCD-IK foot-lock (≤3 mm, 0 penetration) | `footlock.py` | BODY-2 | Foot-Skate & Physics | | TODO |
| ☐ | FOOT-2 | Physics refinement: PhysPT default; PHC/PULSE on MuJoCo+MJX flagship; MultiPhys for doubles | `physics_refine.py` | FOOT-1 | Foot-Skate & Physics | | TODO |

### BALL — Ball tracking + events + 3D physics
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | BALL-1 | TrackNetV3 (→V5) fine-tuned ball tracking + tap-track fallback | `ball_tracknet.py`, `ball_tap_track.py` | CAL-2, DATA-3 | Ball + Events | | TODO |
| ☐ | BALL-2 | Audio "pop" two-stage detector (onset + CNN) + distance-delay correction | `audio_pop.py` | DATA-3 | Ball + Events | | TODO |
| ☐ | BALL-3 | Event fusion (audio + wrist-vel peak + ball inflection) → contact windows; doubles attribution | `event_fusion.py` | BALL-1, BALL-2, BODY-2 | Ball + Events | | TODO |
| ☐ | BALL-4 | 3D ball physics (EKF + RANSAC parabola + z=0 bounce + Magnus + pickleball aero) | `ball_physics3d.py` | BALL-1, CAL-2 | Ball + Events | | TODO |

### RKT — Racket 6DoF
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | RKT-1 | Detect (RTMDet+SAM2) → top/bottom/handle keypoints + corners → GigaPose/FoundPose + grip prior → PnP-IPPE → UKF SE(3) → physics-validate; contact-point + face-normal | `racket6dof.py` | BODY-2, BALL-4, DATA-4 | Racket 6DoF | | TODO |

### MET — Metrics, insights, confidence
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | MET-1 | Biomechanics metrics (foot/NVZ, zones, spacing, balance, X-factor, contact point, velocity w/ tiers) | `movement_metrics.py` | FOOT-1, BALL-3, RKT-1 | Metrics & Insights | | TODO |
| ☐ | MET-2 | Rule-based insight engine + confidence gating | `insight_rules.py`, `confidence.py` | MET-1 | Metrics & Insights | | TODO |

### SHOT — Shot classification + drills
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | SHOT-1 | Shot classifier (BST/PoseConv3D, dataset from audio-snapped+pose) | `shot_classifier.py` | MET-1, DATA-5 | Shot Classification | | TODO |
| ☐ | SHOT-2 | Drill rep counting/verification (wrist-vel peak + contact + state machine) | `drill_verify.py` | MET-1, BALL-3 | Shot Classification | | TODO |

### RPT — Report, LLM copy, visualization
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | RPT-1 | Report artifacts (`habit_report.json`, `coach_report.json`, corrections/exclusion model) | `report_model.py`, `habit_model.py` | MET-2 | Report & Delivery | | TODO |
| ☐ | RPT-2 | LLM coaching copy (latest Claude; facts-in/copy-out; 100% faithfulness gate) | `llm_copy.py` | RPT-1 | Report & Delivery | | TODO |
| ☐ | RPT-3 | Visualization (court map/heatmap + priority metric + self-vs-self) + tiered <10 s delivery | `viz/` | RPT-1 | Report & Delivery | | TODO |

### RPL — 3D replay
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | RPL-1 | Server render bake → one animated scene → export USDZ (OpenUSD) + GLB (pygltflib/smplx); MeshOpt+Draco+KTX2; CDN | `replay_export.py` | FOOT-2, RKT-1, BALL-4 | 3D Replay | | TODO |
| ☐ | RPL-2 | Three.js + R3F web viewer (free-viewpoint, per-point GLB stream, share link) | `viewer/` | RPL-1 | 3D Replay | | TODO |

### DATA — Data & training infrastructure (parallel)
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | DATA-1 | Test-clip dataset (varied camera-height/angle matrix) + label schema | `data/testclips/` | ENV-1 | Test-Clip Spec | Codex | IN-PROGRESS |
| ☐ | DATA-2 | Body pose datasets download + fine-tune pipeline + auto-label/distill loop | `data/pose/`, `scripts/autolabel.py` | DATA-1 | Data Infra | | TODO |
| ☐ | DATA-3 | Ball datasets (Roboflow→x,y) + audio "pop" collection (44.1 kHz) + augmentation | `data/ball/`, `data/audio/` | DATA-1 | Data Infra | | TODO |
| ☐ | DATA-4 | Racket data (RacketVision + synthetic paddle-CAD BlenderProc + ArUco-GT) | `data/racket/` | DATA-1 | Data Infra | | TODO |
| ☐ | DATA-5 | Shot-class dataset (audio-snapped + pose-derived labels) | `data/shots/` | DATA-1, BALL-3 | Data Infra | | TODO |

### EVAL — Validation, gates, CI (spans all)
| ☐ | ID | Task | Owns | Deps | Phase | Owner | Status |
|---|----|------|------|------|-------|-------|--------|
| ☐ | **EVAL-0** | **[GPU] Model variant selection** — benchmark each stage's `TECH_STACK §2.3` candidates on real clips (offline via §1.5 lease; live on device); **render side-by-side comparison videos + get human approval per §1.6 (auto-finalize only if obvious)**; lock the approved variant in `models/MANIFEST.json` (offline=accurate, live=light). Benchmark Fast SAM-3D-Body first. | `models/MANIFEST.json`, `racketsport/eval/bench/`, `runs/eval0/` | DATA-1, ENV-2 | Model Variant Selection | Codex | IN-PROGRESS |
| ☐ | EVAL-1 | Eval harness (`racketsport/eval/`, one evaluator per phase → `metrics.json`) | `racketsport/eval/` | DATA-1 | Cross-cutting | Codex | IN-PROGRESS |
| ☐ | EVAL-2 | Validation Protocols A/B/C/D + physics/racket gates wired to the harness | `racketsport/eval/` | EVAL-1 | Cross-cutting | | TODO |
| ☐ | EVAL-3 | Regression CI (block merge on >2% drop) + corrections-flywheel plumbing | `.github/`, `corrections/` | EVAL-1 | Cross-cutting | | TODO |
| ☐ | EVAL-4 | End-to-end integration on 20-min clips + perf + final acceptance | `pipeline.py` | all VERIFIED | End-to-End | | TODO |

---

## 4. Phase-gate summary (must pass before dependents start)

| Gate | Numeric acceptance (see `IMPLEMENTATION_PHASES.md` for the exact test) |
|---|---|
| Court calibration | overlay matches ≥8/10 clips across ≥4 distinct viewpoints; feet-to-world within per-viewpoint budget |
| Person track | stable IDs ≥90% on 2-player clips after confirmation; 20-min clip throughput target met |
| 3D body | per-stage world-MPJPE target on EMDB2/CalTennis; fine-tune ladder hits ~50–70 mm |
| Foot-skate & physics | foot-slide ≤3 mm; zero floor & inter-player penetration |
| Ball + events | ball F1 ≥0.90 (FP <5%); contact timing ≤±2 frames / ±4 ms (audio) |
| Racket 6DoF | face-angle ≤5° vs ArUco GT; contact-point ±1–3 cm |
| Metrics/insights | metric accuracy vs manual review; confidence calibration; velocity Protocol A/B/C pass |
| Replay | loads fast; free-viewpoint works; coach "looks right" review passes |
| End-to-end | fast-tier first screen <10 s; deep-tier p95 under agreed SLA; gross margin target |

---

## 5. Handoff log (append-only — newest last)

> Format: `[TASK-ID] <agent> — built: <what>; artifacts: <files>; tested: <how/result>; next: <what the next agent needs>`

- [ENV-1] Codex — built: initial `main` repo, Phase 0 Python scaffold, documented module stubs under `threed/racketsport/`, artifact schema registry, ffprobe-based clip metadata probe, test-clip ingest script, GPU eval/train lock helpers, placeholder model manifest, Triton scaffold, and web replay package placeholder; artifacts: `threed/racketsport/`, `tests/racketsport/`, `scripts/racketsport/`, `scripts/gpu-*.sh`, `models/MANIFEST.json`, `serving/triton/README.md`, `web/replay/package.json`; tested: local `.venv/bin/python -m pytest -q` passed (5 tests), local ingest smoke wrote `frames_meta.json`, local GPU-lock fallbacks ran, H100 container `/workspace/pickleball/.venv/bin/python -m pytest ... -q` passed (5 tests), H100 `nvidia-smi` via `scripts/gpu-eval-run.sh` showed `NVIDIA H100 80GB HBM3, 81559 MiB, 0 MiB, 0 %`; next: ENV-2 should fill `models/MANIFEST.json` with real checkpoints/checksums and replace the dry-run SAM-3D-Body benchmark with real inference.
- [ENV-2] Codex — built: H100 model manifest verifier, Fast-SAM isolated conda env installer, Detectron2 CUDA build fixes, warm-run benchmark wrapper, and H100 checkpoint inventory for Fast-SAM/SAM assets, MoGe2, SAM2, existing YOLO, YOLO26m, RTMW-l 384, TrackNetV3, TrackNetV3 InpaintNet, and SAT-HMR; artifacts: `scripts/racketsport/smoke_models.py`, `scripts/racketsport/install_fast_sam_env.sh`, `scripts/racketsport/run_fast_sam_benchmark.sh`, `tests/racketsport/test_smoke_models.py`, `models/MANIFEST.json`; tested: H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (18 tests), H100 `smoke_models.py --check-files-only` verified all 10 `available_on_h100` files and sha256s, Fast-SAM-3D-Body profiler ran on H100 with one warmup + five measured warm runs: average 326.91 ms, std 1.72 ms, min 324.68 ms, max 329.90 ms, peak allocated GPU memory 4718.04 MB; next: `mujoco_mjx` remains pending because it is a runtime package/physics stack rather than a simple checkpoint file, and EVAL-0 still needs real test clips plus comparison artifacts before any model choice is approved.
- [ENV-4] Codex — built: deterministic capture-quality scorer for framing, reprojection, blur, exposure, luminance stability, FPS, shutter, shake, and ARKit tracking state; artifacts: `threed/racketsport/capture_quality.py`, `tests/racketsport/test_capture_quality.py`; tested: local `.venv/bin/python -m pytest tests/racketsport` passed (15 tests before DATA-1 merge, then 18 tests after); next: wire real frame/audio QC extraction into ingest so the scorer is fed from video probes instead of sidecar/server signals only.
- [DATA-1] Codex + Helmholtz — built: CPU-only test-clip readiness manifest and CLI validator for required label files from the Test-Clip Dataset Spec; artifacts: `threed/racketsport/testclips.py`, `scripts/racketsport/validate_testclips.py`, `tests/racketsport/test_testclips.py`; tested: local `.venv/bin/python -m pytest tests/racketsport` passed (18 tests), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (18 tests); next: collect or mount real `data/testclips/` clips and labels, then run `scripts/racketsport/validate_testclips.py` until it exits 0.
- [CAL-1] Codex — built: regulation pickleball/tennis court templates, schema-backed zone artifacts, and vertical net-plane artifacts in the shared court world frame; artifacts: `threed/racketsport/court_templates.py`, `threed/racketsport/court_zones.py`, `threed/racketsport/net_plane.py`, `tests/racketsport/test_court_geometry.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_court_geometry.py -q` passed (6 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (24 tests); next: CAL-2 can consume `get_court_template()`, `build_court_zones()`, and `build_net_plane()` for solvePnP world points, overlay zones, and net references.
- [ENV-4] Codex — built: sampled ffmpeg clip-QC probe wired into test-clip ingest, including pure-Python blur/luminance metrics, QC decode FPS, and `capture_quality` output in each `frames_meta.json`; artifacts: `threed/racketsport/io_decode.py`, `scripts/racketsport/ingest_testclips.py`, `tests/racketsport/test_io_decode.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_io_decode.py -q` passed (5 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (27 tests); next: replace/augment this CPU sampled path with true H100 NvDEC iteration and measure the Phase 0 decode throughput gate on real 1080p/4K test clips.
- [ENV-3] Codex — built: iOS Swift package module scaffold, shared `CaptureSidecar` Codable contract, minimal SwiftUI app entry point, and Xcode project/scheme skeleton; artifacts: `ios/Package.swift`, `ios/README.md`, `ios/App/`, `ios/Core/`, `ios/Capture/`, `ios/Calibration/`, `ios/FastTier/`, `ios/Guidance/`, `ios/Upload/`, `ios/Replay/`, `ios/SwayBody.xcodeproj/`; tested: `swift package --package-path ios describe` passed, `swift test --package-path ios` passed (2 tests), `plutil -lint ios/SwayBody.xcodeproj/project.pbxproj ios/App/Info.plist` passed, `xmllint --noout` passed for the shared scheme/workspace XML; blocked: any `xcodebuild` command fails before project evaluation because `/Applications/Xcode.app/Contents/Frameworks/IDESimulatorFoundation.framework` cannot load a `DVTDownloads` symbol, `xcodebuild -runFirstLaunch` hung for >2 minutes, and passwordless sudo is unavailable; next: repair/reinstall Xcode or run the admin first-launch flow, then rerun `xcodebuild -project ios/SwayBody.xcodeproj -scheme SwayBody -sdk iphonesimulator -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build` before moving ENV-3 to DONE.
- [ENV-4] Codex — built: explicit ffmpeg decode-throughput benchmark helper with CPU/CUDA backends for Phase 0 ingest validation; artifacts: `threed/racketsport/io_decode.py`, `scripts/racketsport/benchmark_decode.py`, `tests/racketsport/test_io_decode.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_io_decode.py -q` passed (7 tests), local CLI smoke decoded a generated 10-frame clip, H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport/test_io_decode.py -q` passed (7 tests), H100 ffmpeg advertised CUDA/CUVID decoders (`h264_cuvid`, `hevc_cuvid`) and CUDA filters (`scale_cuda`, `hwupload_cuda`), H100 synthetic 5 s 1080p60 H.264 decode benchmark measured CPU 1626 fps / 27.1x realtime and CUDA 307 fps / 5.1x realtime; next: run the benchmark on real 20-min 1080p/4K test clips and pick the backend empirically instead of assuming CUDA wins.
- [ENV-2] Codex — built: isolated MuJoCo/MJX runtime env installer and smoke test for physics-stack availability on the H100; artifacts: `scripts/racketsport/install_mujoco_mjx_env.sh`, `scripts/racketsport/smoke_mujoco_mjx.py`, `models/MANIFEST.json`; tested: H100 `/opt/conda/envs/racketsport_mjx/bin/python /tmp/mjx_smoke.py` passed with `jax_version=0.10.2`, `mujoco_version=3.10.0`, device `cuda:0`, and a 5-step MJX sphere simulation; next: FOOT-2 still needs task-specific physics integration and validation, but `mujoco_mjx` is no longer pending as an installed runtime package.
- [ENV-2] Codex — built: completed the current H100 checkpoint manifest inventory with Multi-HMR2.b plus MMPose RTMW-x 384, RTMPose Body26 m/l/x 384, and RTMPose COCO-WholeBody m 256 + l/x 384 entries; artifacts: `models/MANIFEST.json`, `tests/racketsport/test_model_manifest.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_model_manifest.py tests/racketsport/test_manifest_report.py -q` passed (4 tests), local manifest report shows 18 `available_on_h100` checkpoint files plus 1 `available_runtime_on_h100` runtime and 0 pending/missing entries, H100 `/opt/conda/envs/fast_sam_3d_body/bin/python scripts/racketsport/smoke_models.py --manifest models/MANIFEST.json --check-files-only --json` verified all 18 file-backed entries with 0 failures; next: EVAL-0 must benchmark candidate variants on real clips and render comparison artifacts before any default model choice is approved.
- [CAL-2] Codex — built: CPU-safe sidecar/manual-tap calibration slice with sidecar intrinsics loading, net-center court-template corner correspondences, pure-Python world-to-image homography, reprojection median/p95 gate constants, sidecar capture-quality merge, static multi-frame tap averaging, and a calibration CLI that writes `court_calibration.json`, `court_zones.json`, and `net_plane.json`; artifacts: `threed/racketsport/court_calibration.py`, `scripts/racketsport/calibrate.py`, `tests/racketsport/test_court_calibration.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_court_calibration.py -q` passed (8 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (37 tests), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport/test_court_calibration.py -q` passed (8 tests), and H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (37 tests); next: add real OpenCV `solvePnP` pose refinement, overlay rendering, drift guard, and run Phase 1 gates on real labeled test clips before marking CAL-2 done.
- [CAL-2] Codex — built: optional OpenCV `solve_camera_pose()` helper for full 6-DoF PnP refinement from world/image correspondences plus sidecar intrinsics, returning schema-compatible `CourtExtrinsics` and camera height from the solved camera center; artifacts: `threed/racketsport/court_calibration.py`, `tests/racketsport/test_court_calibration.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_court_calibration.py -q` passed with the OpenCV-specific test skipped because local `.venv` lacks `cv2`/`numpy`, H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport/test_court_calibration.py -q` passed (9 tests, including solvePnP); next: wire solvePnP into the calibration CLI with confidence/fallback policy, then add overlay rendering and drift guard on real labeled clips.
- [CAL-2] Codex — built: wired solved OpenCV PnP extrinsics into `calibration_from_manual_taps()` / multi-frame artifact creation with fallback to ARKit sidecar seed when `cv2`/`numpy` are unavailable or the solve fails; artifacts: `threed/racketsport/court_calibration.py`, `tests/racketsport/test_court_calibration.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_court_calibration.py -q` passed (8 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport/test_court_calibration.py -q` passed (10 tests, including solved-pose preference); next: add overlay rendering, drift guard, explicit solver-status metadata if the schema expands, and run Phase 1 gates on real labeled clips.
- [CAL-2] Codex — built: deterministic drift guard for calibrated court points, including scheduled frame checks and p95 reprojection drift detection against the 15 px Phase 1 gate; artifacts: `threed/racketsport/drift_guard.py`, `tests/racketsport/test_drift_guard.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_drift_guard.py -q` passed (3 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (40 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (42 tests); next: feed observed points from ORB/optical-flow court-line tracking and validate injected 20 px bumps on real clips.
- [CAL-2] Codex — built: world-point zone classification for pickleball and tennis with specific-zone priority before broad court fallback, supporting later court-polygon filtering and metric/NVZ rules; artifacts: `threed/racketsport/court_zones.py`, `tests/racketsport/test_court_geometry.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_court_geometry.py -q` passed (8 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (42 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (44 tests); next: use `classify_point()` in TRK-1/MET rules after CAL-2 produces real calibrated world points.
- [CAL-2] Codex — built: pure-Python world-point projection from solved calibration extrinsics/intrinsics and regulation net-plane endpoint projection for overlay/net-line cross-checks; artifacts: `threed/racketsport/court_calibration.py`, `threed/racketsport/net_plane.py`, `tests/racketsport/test_court_calibration.py`, `tests/racketsport/test_court_geometry.py`; tested: local `.venv/bin/python -m pytest tests/racketsport -q` passed (44 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (46 tests); next: use projected net endpoints in overlay rendering and compare against labeled/visible net lines on real clips.
- [TRK-1] Codex — built: CPU-only tracking primitives for calibrated court filtering and identity locking: typed person detections with foot world points, court-polygon filtering through CAL-2 zone geometry, confidence-ranked N-lock, and metric ground-step plausibility for teleport rejection; artifacts: `threed/racketsport/person_fast.py`, `threed/racketsport/track_lock.py`, `tests/racketsport/test_tracking_primitives.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_tracking_primitives.py -q` passed (4 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (48 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (50 tests); next: integrate YOLO26m detections/BoT-SORT outputs into these primitives and evaluate on real clips once CAL-2 has verified world points.
- [TRK-1] Codex — built: deterministic doubles identity helpers for court-side/lateral role assignment from world positions plus one-tap coach anchor to bind a semantic label to the nearest locked track within a metric radius; artifacts: `threed/racketsport/doubles_id.py`, `tests/racketsport/test_tracking_primitives.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_tracking_primitives.py -q` passed (6 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (50 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (52 tests); next: combine with tracker outputs and color/ReID recovery once real clip tracks exist.
- [ENV-4] Codex — tested: H100 copied existing smoke clip `/Users/arnavchokshi/Desktop/CV_pipeline/sam4dbody/tmp/pickleball_body4d_smoke/source_vQhtz8l6VqU.mp4` into `/workspace/pickleball/data/testclips/smoke/` and ran `scripts/racketsport/benchmark_decode.py` on real 12 s 1280x720 H.264/25 fps video; CPU decode measured 2040 fps / 81.6x realtime and CUDA decode measured 416.6 fps / 16.7x realtime; next: still need representative 1080p/4K 20-min clips before selecting a default decode backend.
- [EVAL-0] Codex — tested: extracted `/workspace/pickleball/runs/phase0/real_smoke/frame_001.jpg` from the same 12 s pickleball smoke clip and ran `scripts/racketsport/run_fast_sam_benchmark.sh runs/phase0/fast_sam_real_smoke_profile` with `FAST_SAM_WARMUP_RUNS=1`, `FAST_SAM_BENCHMARK_RUNS=3`, and `FAST_SAM_IMAGE_PATH` pointing at that frame; H100 Fast-SAM-3D-Body detected 4 people and measured average 342.35 ms, std 2.50 ms, min 340.38 ms, max 345.87 ms, peak allocated GPU memory 4720.29 MB; next: EVAL-0 still needs representative labeled real clips, side-by-side artifacts, and the human approval gate before model choice is approved.
- [TRK-1] Codex — built: detector-box to ground-plane association bridge using calibrated homography inversion, including bottom-center footpoint conversion from image-space bboxes into court world coordinates before court-polygon filtering; artifacts: `threed/racketsport/court_calibration.py`, `threed/racketsport/person_fast.py`, `tests/racketsport/test_court_calibration.py`, `tests/racketsport/test_tracking_primitives.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_court_calibration.py tests/racketsport/test_tracking_primitives.py -q` passed (17 passed, 2 skipped), local `.venv/bin/python -m pytest tests/racketsport -q` passed (52 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (54 tests); next: connect this adapter to real detector/tracker outputs and evaluate on labeled clips.
- [EVAL-0] Codex — built: machine-readable Fast-SAM benchmark recorder that parses `profile_nsight.py` stdout into `sam3dbody_benchmark.json`, plus wrapper teeing to `profile_stdout.log` before writing metrics and normalizing relative output dirs before `cd`; artifacts: `scripts/racketsport/benchmark_sam3dbody.py`, `scripts/racketsport/run_fast_sam_benchmark.sh`, `tests/racketsport/test_benchmark_sam3dbody.py`, `tests/racketsport/test_shell_scripts.py`, `/workspace/pickleball/runs/phase0/fast_sam_real_smoke_json_check2/`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_benchmark_sam3dbody.py tests/racketsport/test_shell_scripts.py -q` passed (5 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (56 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (58 tests), H100 real-smoke wrapper wrote `profile_stdout.log` and `sam3dbody_benchmark.json` with status `measured`, 4 detected people, average 350.07 ms over one measured run, and peak VRAM 4720.29 MB; next: EVAL-0 still needs representative labeled clips, side-by-side variant artifacts, and human/obvious-winner approval before any model choice is locked.
- [DATA-1] Codex — built: optional `clip_metadata.json` schema and coverage-matrix accounting for the required 24-clip varied-camera test set, including counts and gap messages for camera height/angle, play type, environment, frame rate, length, and ArUco racket-GT coverage without changing label-readiness semantics; the validator now exposes `dataset_ready` and exits 0 only when both labels and the dataset matrix are complete; artifacts: `threed/racketsport/testclips.py`, `scripts/racketsport/validate_testclips.py`, `tests/racketsport/test_testclips.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_testclips.py -q` passed (6 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (59 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (61 tests), H100 real `validate_testclips.py --root /workspace/pickleball/data/testclips` exited 1 as expected with `total_clips=1`, `ready_clips=0`, `metadata_ready_clips=0`, `dataset_ready=false`, and the 24-clip matrix gaps listed; next: add actual collected clips/metadata under `data/testclips/` and drive `coverage_gaps` to empty.
- [EVAL-1] Codex — built: CPU-only phase-eval scaffold with shared `PhaseEvalMetrics` schema/registry entry, JSON writer helpers, and first Phase 1 calibration evaluator that consumes DATA-1 label readiness plus `court_calibration.json`, `court_zones.json`, and `net_plane.json` run artifacts to write `runs/phase1/metrics.json`; artifacts: `threed/racketsport/schemas/__init__.py`, `threed/racketsport/eval/metrics.py`, `threed/racketsport/eval/calib_eval.py`, `tests/racketsport/test_eval_metrics.py`, `tests/racketsport/test_schemas.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_eval_metrics.py tests/racketsport/test_schemas.py -q` passed (7 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (63 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (65 tests), H100 `python -m threed.racketsport.eval.calib_eval --root /workspace/pickleball/runs/phase1 --labels /workspace/pickleball/data/testclips --out /workspace/pickleball/runs/phase1/metrics.json` wrote ignored `metrics.json` with `status=not_measured`, `total_clips=1`, `ready_clips=0`, and `evaluated_clips=0` as expected for the current incomplete smoke dataset; next: run calib_eval against real DATA-1 clips, then add track/body/ball/racket evaluator modules and wire EVAL-2 numeric gates.
- [EVAL-1] Codex — built: Phase 2 `track_eval` scaffold that consumes DATA-1 label readiness plus `tracks.json`, writes `runs/phase2/metrics.json`, validates the `Tracks` artifact, and records initial artifact-readiness/player-count/frame-count metrics while leaving IDF1, ID switches, spectator rejection, and throughput gates for EVAL-2; artifacts: `threed/racketsport/eval/track_eval.py`, `tests/racketsport/test_eval_metrics.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_eval_metrics.py -q` passed (5 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (65 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (67 tests), H100 `python -m threed.racketsport.eval.track_eval --root /workspace/pickleball/runs/phase2 --labels /workspace/pickleball/data/testclips --out /workspace/pickleball/runs/phase2/metrics.json` wrote ignored `metrics.json` with `status=not_measured`, `total_clips=1`, `ready_clips=0`, and `evaluated_clips=0` as expected for the current incomplete smoke dataset; next: add body/ball/racket evaluator modules and numeric EVAL-2 gates.
- [EVAL-1] Codex — built: Phase 3 `body_eval` scaffold that consumes DATA-1 label readiness plus `smpl_motion.json` and `skeleton3d.json`, writes `runs/phase3/metrics.json`, validates `SmplMotion`/`Skeleton3D`, and records initial SMPL player/frame plus preview skeleton counts while leaving MPJPE, foot/NVZ, angle, spacing, and FPS gates for EVAL-2; artifacts: `threed/racketsport/eval/body_eval.py`, `tests/racketsport/test_eval_metrics.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_eval_metrics.py -q` passed (7 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (67 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (69 tests), H100 `python -m threed.racketsport.eval.body_eval --root /workspace/pickleball/runs/phase3 --labels /workspace/pickleball/data/testclips --out /workspace/pickleball/runs/phase3/metrics.json` wrote ignored `metrics.json` with `status=not_measured`, `total_clips=1`, `ready_clips=0`, and `evaluated_clips=0` as expected for the current incomplete smoke dataset; next: add ball/racket evaluator modules and numeric EVAL-2 gates.
- [EVAL-1] Codex — built: Phase 4 `physics_eval` scaffold that consumes DATA-1 label readiness plus refined `smpl_motion.json`, writes `runs/phase4/metrics.json`, validates `SmplMotion`, and records initial player/frame, foot-contact-frame, skate-free-player, physics-mode, and GRF-frame counts while leaving foot-slide, floor penetration, inter-player penetration, contact precision/recall, and acceleration-jitter gates for EVAL-2; artifacts: `threed/racketsport/eval/physics_eval.py`, `tests/racketsport/test_eval_metrics.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_eval_metrics.py -q` passed (14 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (75 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (77 tests), H100 `python -m threed.racketsport.eval.physics_eval --root /workspace/pickleball/runs/phase4 --labels /workspace/pickleball/data/testclips --out /workspace/pickleball/runs/phase4/metrics.json` wrote ignored `metrics.json` and exited 1 with `status=not_measured`, `total_clips=1`, `ready_clips=0`, and `evaluated_clips=0` as expected for the current incomplete smoke dataset; next: wire EVAL-2 numeric gates and dashboard/CI.
- [EVAL-1] Codex — built: Phase 5 `ball_event_eval` scaffold that consumes DATA-1 label readiness plus `ball_track.json` and `contact_windows.json`, writes `runs/phase5/metrics.json`, validates `BallTrack`/`ContactWindows`, and records initial ball-frame/contact/bounce counts while leaving ball F1, contact timing, bounce/net accuracy, and physics-fit residuals for EVAL-2; artifacts: `threed/racketsport/eval/ball_event_eval.py`, `tests/racketsport/test_eval_metrics.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_eval_metrics.py -q` passed (9 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (69 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (71 tests), H100 `python -m threed.racketsport.eval.ball_event_eval --root /workspace/pickleball/runs/phase5 --labels /workspace/pickleball/data/testclips --out /workspace/pickleball/runs/phase5/metrics.json` wrote ignored `metrics.json` and exited 1 with `status=not_measured`, `total_clips=1`, `ready_clips=0`, and `evaluated_clips=0` as expected for the current incomplete smoke dataset; next: add racket evaluator module and numeric EVAL-2 gates.
- [EVAL-1] Codex — built: Phase 6 `racket_eval` scaffold that consumes DATA-1 label readiness plus `racket_pose.json`, writes `runs/phase6/metrics.json`, validates `RacketPose`, and records initial racket-player/frame/contact counts while leaving face-angle, contact-point, ArUco GT, and physics-validation gates for EVAL-2; artifacts: `threed/racketsport/eval/racket_eval.py`, `tests/racketsport/test_eval_metrics.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_eval_metrics.py -q` passed (11 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (71 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (73 tests), H100 `python -m threed.racketsport.eval.racket_eval --root /workspace/pickleball/runs/phase6 --labels /workspace/pickleball/data/testclips --out /workspace/pickleball/runs/phase6/metrics.json` wrote ignored `metrics.json` and exited 1 with `status=not_measured`, `total_clips=1`, `ready_clips=0`, and `evaluated_clips=0` as expected for the current incomplete smoke dataset; next: wire EVAL-2 numeric gates and dashboard/CI.
- [EVAL-1] Codex — built: Phase 7 `metric_eval` scaffold that consumes DATA-1 label readiness plus `racket_sport_metrics.json` and draft `habit_report.json`, writes `runs/phase7/metrics.json`, validates `RacketSportMetrics`/`HabitReport`, and records initial player, shot, metric-value, gated-metric, habit, and coverage counts while leaving metric accuracy, rule precision, confidence calibration, and rotation-gating gates for EVAL-2; artifacts: `threed/racketsport/eval/metric_eval.py`, `tests/racketsport/test_eval_metrics.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_eval_metrics.py -q` passed (16 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (77 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (79 tests), H100 `python -m threed.racketsport.eval.metric_eval --root /workspace/pickleball/runs/phase7 --labels /workspace/pickleball/data/testclips --out /workspace/pickleball/runs/phase7/metrics.json` wrote ignored `metrics.json` and exited 1 with `status=not_measured`, `total_clips=1`, `ready_clips=0`, and `evaluated_clips=0` as expected for the current incomplete smoke dataset; next: add shot/drill, copy, replay, and e2e evaluators.
- [EVAL-1] Codex — built: Phase 8 `shot_drill_eval` scaffold that consumes DATA-1 label readiness plus `racket_sport_metrics.json` and `drill_report.json`, writes `runs/phase8/metrics.json`, validates `RacketSportMetrics`/`DrillReport`, and records initial shot, shot-type, drill-rep, clean-rep, and fault-rep counts while leaving shot macro-F1, top-2 accuracy, and rep-count error gates for EVAL-2; artifacts: `threed/racketsport/eval/shot_drill_eval.py`, `tests/racketsport/test_eval_metrics.py`; tested: local `.venv/bin/python -m pytest tests/racketsport/test_eval_metrics.py -q` passed (18 tests), local `.venv/bin/python -m pytest tests/racketsport -q` passed (79 passed, 2 skipped), H100 `/opt/conda/envs/fast_sam_3d_body/bin/python -m pytest tests/racketsport -q` passed (81 tests), H100 `python -m threed.racketsport.eval.shot_drill_eval --root /workspace/pickleball/runs/phase8 --labels /workspace/pickleball/data/testclips --out /workspace/pickleball/runs/phase8/metrics.json` wrote ignored `metrics.json` and exited 1 with `status=not_measured`, `total_clips=1`, `ready_clips=0`, and `evaluated_clips=0` as expected for the current incomplete smoke dataset; next: add copy, replay, and e2e evaluators.

---

## 6. Decisions log (append-only)

> Record any model swap, deviation, assumption, or resolved ambiguity so other agents see it.

- Stack baseline as of handoff: Fast SAM-3D-Body backbone + our world-grounding; YOLO26m + BoT-SORT-ReID; foot-lock to Z=0 + PhysPT/MuJoCo; racket PnP-IPPE; TrackNetV3→V5; RealityKit/USDZ native + Three.js/GLB web. Fallbacks: Fast SAM-3D-Body→original→NLF; YOLO26→YOLO11; SAT-HMR↔Multi-HMR 2; PhysPT→PHC. Single-camera product; multi-cam future; multi-view training-only.
- 2026-06-26 Codex: canonical server filenames are the `IMPLEMENTATION_PHASES.md §0.2` names (`person_fast.py`, `audio_pop.py`, `footlock.py`, `racket6dof.py`, etc.). Older MVP shorthand names such as `person_fasttier.py`, `audio_events.py`, `foot_lock.py`, and `racket_pose6dof.py` are retired aliases and should not be implemented.
- 2026-06-26 Codex: external existence check found official Ultralytics YOLO26 docs/assets and public Fast SAM-3D-Body project/GitHub pages. This verifies the names exist, not our target H100 performance, checkpoint access, or commercial license posture.
- 2026-06-26 Codex: CAL-1 uses a shared court world frame with origin at net center, +x across court width, +y toward the far baseline, +z up, and meters for artifact coordinates. Regulation dimensions are stored in feet/inches and converted at the API boundary: pickleball 20x44 ft, 7 ft NVZ, 22 ft net, 34/36 in net heights; tennis 78 ft length, 36 ft doubles width, 27 ft singles width, 21 ft service line, 42 ft net between posts, 36/42 in net heights.
- 2026-06-26 Codex: Fast-SAM-3D-Body must use the isolated `/opt/conda/envs/fast_sam_3d_body` env on the H100. Detectron2 builds successfully there only after installing CUDA 12.4 toolkit plus conda-forge GCC/G++ 13 and setting `CUDAHOSTCXX`; GCC 14 is rejected by CUDA 12.4.
- 2026-06-26 Codex: Fast-SAM-3D-Body H100 smoke benchmark is successful but not variant approval. Measured on one sample image with one detected person, YOLO11n detector, MoGe2 FOV, no SAM2 segmentation: average 326.91 ms over five warm runs and 4718.04 MB peak allocated GPU memory. EVAL-0 still requires real test clips, side-by-side artifacts, and the §1.6 approval gate.
- 2026-06-26 Codex: ENV-4 now emits clip-QC and `capture_quality` from sampled ffmpeg frames as the deterministic local fallback. This validates scoring and metadata plumbing, but it is not the Phase 0 NvDEC performance gate; true H100 NvDEC decode throughput still needs implementation and measurement on real clips.
- 2026-06-26 Codex: ENV-3 uses checklist ownership folders (`ios/Capture/`, `ios/Calibration/`, etc.) with Swift package target names (`SwayCapture`, `SwayCalibration`, etc.) so task ownership stays aligned with `BUILD_CHECKLIST.md` while import/module names stay idiomatic. The Xcode project uses Xcode 26 file-system-synchronized app sources plus a local package reference to `ios/Package.swift`.
- 2026-06-26 Codex: H100 container ffmpeg exposes CUDA/CUVID decode support, but a short synthetic 1080p60 benchmark made CPU decode faster than CUDA because CUDA startup/copy overhead dominates that clip. ENV-4 decode backend should be selected from real clip measurements; do not hardcode CUDA as always faster.
- 2026-06-26 Codex: MuJoCo/MJX is installed as an isolated runtime env at `/opt/conda/envs/racketsport_mjx` with `jax[cuda12]` and `mujoco-mjx[warp]`, following the official package guidance. Keep it separate from the Fast-SAM env unless a later integration task deliberately consolidates environments.
