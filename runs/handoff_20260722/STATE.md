# SESSION HANDOFF — 2026-07-22 ~10:30 PDT

Written for a **fresh session with zero conversational context** (owner is switching accounts).
Read this, then `runs/regroup_20260721/EXACT_PLAN.md`, then `OWNER_CHECKIN.md`.
Assume nothing here is "obvious" — it was all learned expensively.

---

## 1. WHAT THIS IS

DinkVision: single static-camera iPhone pickleball app → 3D replay (court, 4 players with meshes,
ball flight/bounces, in/out) + evidence-linked coaching. `VERIFIED=0` is binding: **no capability
is promoted**; every number below is a measurement, not a claim of product readiness.

Authorities, in order: `CLAUDE.md` → `AGENTS.md` → `NORTH_STAR_ROADMAP.md` (sole product/truth/
sequencing authority) → `runs/regroup_20260721/EXACT_PLAN.md` (the current 5-day data-adjudication
sprint that is being executed) → `RUNBOOK.md` for commands. `OWNER_CHECKIN.md` is the standing
always-current owner-facing file (to-dos + best-results table); update it at every landing.

## 2. HOW WORK IS ORGANIZED (owner-directed structure, 2026-07-21)

- **Main session** = coordinator only. Stays light, relays to owner, makes go/no-go calls.
- **Orchestrator subagent** = executes EXACT_PLAN, delegates all coding. Predecessor died 07:10
  PDT (Fable-5 usage limit); successor running on Opus (`af7e342733fc5c0f6`) was told to reach a
  stopping point and write `runs/handoff_20260722/ORCHESTRATOR_STATE.md` — **read that file too**,
  it has the lane-level detail this file summarizes.
- **Codex lanes** do the coding: `codex exec --cd <repo> --sandbox workspace-write -c
  model="gpt-5.6-sol" -c model_reasoning_effort=<high|xhigh|ultra> --output-schema
  docs/racketsport/lane_report.schema.json -o <lane>/report.json < <lane>/spec.md > <lane>/log.txt`
  (nohup-detached, absolute paths, background).
- **Sonnet subagents** do GPU/SSH/network work (Codex has no network in sandbox).

## 3. STANDING POLICIES (non-negotiable — each traces to a specific past failure)

1. **gpt-5.6-sol ULTRA reviews everything before any commit.** Round-1 rejection is the norm, not
   the exception: on 2026-07-21, 5 of 5 lanes were rejected with reproduced defects.
2. **No judge peeking.** Never tune after seeing a frozen score. (A lane once went 0.6767 → peek →
   0.7800; work was discarded and that component's judges retired.)
3. **Data gates before GPU.** No VM starts before its named data gate passes.
4. **Every lane ends in a measured number or a named negative.**
5. **GPU money discipline**: spot VMs, boot-armed `shutdown -P +N` rails, idle watchdogs, mandatory
   teardown, list-confirm, report spend. Never touch a VM owned by a live agent without messaging
   that agent first. **Disks survive VM deletion and keep billing** — rule on them explicitly.
6. **Anti-passive-wait**: an agent that ends its turn "waiting for X" is dead. Poll with bounded
   foreground loops.
7. **Everything-helps-everything**: cross-signal fusion is the moat (wrists→hits, hits+joints→ball
   3D, court anchors everything). Every lane states what it consumes and feeds.

## 4. MEASURED TRUTH AS OF NOW (all frozen-protocol, independent held-out data)

| Capability | Number | Protocol | Status |
|---|---|---|---|
| BALL detector baseline | pooled **F1@20 0.5670** (indoor HyU **0.7395** / outdoor-night Ezz **0.2933**) | official WASB zero-shot on the owner-attested 167-row source-held judge | measured, cross-verified by 2 codepaths |
| BALL judge | 167 rows (94 pos / 73 owner-attested neg), 0 contamination vs 2,953 protected frames | 3 ultra review rounds, byte-bound to export sha256 | ACCEPTED, frozen |
| BALL retrain (B2) | **not run** | — | blocked: B1 data build timed out at 5/7 videos + B2 unauthorized (see §6) |
| EVENT head | A (owner-61 only) **0.0**; B (+pb.vision teacher) **0.1304**; C (placebo, shuffled times) **0.0**; timing p90 64→2 frames | frozen A/B/C, owner-41 val, 1000 steps each | **EVENT_PBV_SEED1_NO_LIFT** — closed. B failed guards: negFP 4/22 (allowed 2), rate 0.107/s (band 0.3-1.0) |
| PERSON (Roboflow-only) | 8,887 imgs / **7** original-footage families vs floor of 8 | pixel-level leak scan, 66.4M pairs, 0 leaks | **PERSON_RF_POOL_TOO_THIN** — verified negative, $0 GPU |
| PERSON (mixed-pool, owner-directed) | hemel_test F1 **0.663** vs control 0.617 (WIN, all metrics); od8al_val F1 **0.735** vs 0.819 (LOSS, precision 0.663 vs 0.856) | held-out human families only, identical conf/NMS | **HONEST MISS** — "both families non-negative" bar not met; no post-hoc tuning |
| COURT | 66 usable rows / 18 families staged; adapter accepted (4 rounds) | fail-closed denial, family grouping, geometry validity | **ADAPTER_READY_AWAITING_EXPORT** (owner labels) |
| ReID | — | — | **NO_ATTEMPT**; displaced by owner's mixed-pool directive; top queue item |

**The one-sentence diagnosis driving the whole sprint:** every honest failure is domain/venue
generalization, and the binding constraint is venue-diverse *training* data — which we own
(pb.vision 13 videos fully licensed, YouTube harvest, owner labels) but had never properly queued.

## 5. CLOUD + LOCAL STATE (verified 10:20 PDT)

- **VMs**: `pickleball-gpu-person`, `pickleball-gpu-ball-f`, `pickleball-a100-fleet1` — ALL
  TERMINATED. Zero compute billing.
- **Disks (STILL BILLING, ~$20/mo each, 200GB)**: `pickleball-gpu-person` (us-central1-a),
  `pickleball-gpu-ball-disk-f` (us-central1-f), `pickleball-a100-fleet1` (asia-southeast1-a).
  Orchestrator was ordered to rule keep-vs-delete; **verify this was done**. Standing read:
  ball-f disk holds the 5/7 completed B1 artifacts (~6 GPU-hours of work) → KEEP until B1
  completes; person disk artifacts are pulled + 62/62 verified → deletable; fleet1 is an old
  prior-wave disk → verify then delete.
- **Local**: no Codex lanes running at time of writing (orchestrator may have dispatched since).
  CVAT Docker is **OFF at owner's request** (disk space) — annotations are safe in Docker volumes.
- **Spend**: ~$40-50 total across 2026-07-21/22 (ABC ~$20, ball ~$7-9, person ~$5-7, baseline ~$1).
- **git**: HEAD `e1e2184d` on `main`, pushed. 33 modified/untracked paths in the working tree,
  mostly large generated artifacts and pre-existing deletions — inspect before any bulk `git add`.

## 6. OPEN THREADS — EXACT NEXT ACTIONS

### A. BALL (critical path, best near-term win)
1. **Add resume support to `scripts/racketsport/build_pbvision_ball_sst.py`** (idempotent
   per-video artifact detection + hash verification, so a resumed run consumes the 5 completed
   videos and builds only the remaining 2). CPU-only, cheap, no GPU. Ultra-review it.
2. **Resume B1** on the ball-f disk with a rail sized from **measured** per-video rates (the 5/7
   evidence has real timings). Do NOT reuse EXACT_PLAN's "0.5-1 GPU-hour" estimate — it was wrong
   by ~6x; the 7 videos are ~83 content-minutes including one 4K and one 60fps source.
3. **B2 AUTHORIZATION GAP (do not skip):** `runs/lanes/ball_b1b2_prep_20260721_review/review_r3.json`
   → `GPU_DISPATCH_DECISION.decision` reads verbatim
   `"DISPATCH_B1_AND_CUDA_PARITY_AFTER_PREFLIGHT; DO_NOT_ARM_B2_YET"`. A Sonnet lane correctly
   refused to arm B2 on a peer's looser paraphrase. **B2 requires a fresh review_r4-equivalent
   dispatch decision explicitly superseding that string.** Obtain it in parallel so B2 fires the
   moment B1's gate passes.
4. Bars are frozen: A-arm (human-only retrain) must reach pooled **≥0.6170** with both sources
   non-negative; B (with pb.vision teacher) must then beat A by **≥0.03**.

### B. PERSON — rule on the mixed-pool miss
The loss is a **precision collapse on one family**, not a uniform regression. Cheap, preregistered,
separately-scored hypotheses (test WITHOUT peeking at the frozen judge): teacher-confidence
threshold too permissive on out-of-domain footage; per-venue exposure imbalance; self-training
reinforcing the teacher's own errors on the family nearest its training distribution. **Do not
re-run the same experiment hoping for a better roll.** The strategic alternative: build a
pb.vision human-label person pack (model pre-labels → owner confirms/rejects fast) — the owner's
182-image audit already established Roboflow's *label quality* is fine; *footage diversity* is the
defect.

### C. COURT — blocked only on owner
CVAT tasks 88-91 (100 frames, 28 different YouTube videos, ~45-60 min). Everything downstream is
built and reviewed. On export: file to
`cvat_upload/exports/court_diversity_20260712/{shard_name}_annotations.zip`, then C1 runs.

### D. EVENT — closed, with a justified successor
E1 is spent; protected-50 one-touch token **UNUSED**. The causal split (teacher timing lifts,
placebo doesn't) is the first evidence that this head *learns event timing when given enough
correct timestamps*. That converts "owner labels more events" from hope into a justified design —
but it needs a **separately registered** experiment, not a re-score of E1.

### E. ReID — `NO_ATTEMPT`, top of the Day-3 queue unless re-prioritized.

## 7. OWNER CONTEXT (read before writing to them)

- Types fast with typos — interpret intent, don't nitpick.
- **Frustrated by process overhead and slow visible progress.** Wants REAL measured results,
  maximum parallelism, no idle spend. Under-claim and deliver; never predict results.
- They are right that gates cost time, and also right that gates exist for a reason. The answer is
  never to weaken a gate — it's to make the work between gates faster and more parallel.
- pb.vision data is **fully licensed** (signed agreement, training + commercial). Use it maximally.
  Compare-only IDs `83gyqyc10y8f`, `iottnc0h3ekn`, `o4dee9dn0ccr` remain quarantined by protocol.
- Static camera is a confirmed v1 requirement; calibrations pool the ENTIRE video, never one frame.
- Owner labeling is the scarce resource. Ask for it in small, well-specified batches with dead
  simple instructions, and never ask twice for the same thing.

## 8. GOTCHAS THAT COST US TIME (don't rediscover)

- **Disks outlive VMs and keep billing.** Rule on them at teardown.
- **Boot rails kill long builds**: size the rail from *measured* throughput, and give any
  long-running builder resume support BEFORE running it (this cost ~6 GPU-hours).
- **Spot stockouts happen** (A100 in us-central1-a): relocate via disk snapshot to another zone.
- **Fresh-clone-on-VM discipline caught a real contamination attempt** — VMs must run committed
  code only; uncommitted working-tree code physically can't leak in.
- **Peer-agent claims must be independently verified** (a peer's paraphrase of an authorization was
  looser than the authoritative JSON; a peer also made false claims about another agent's actions).
- **Monitor greps false-fire** on spec text; read the actual final message before acting.
- **`gcloud auth` dies periodically**: raw `ssh -i ~/.ssh/google_compute_engine arnavchokshi@<IP>`
  bypasses it for existing VMs; create/delete needs the owner's interactive `gcloud auth login`.
- **Laptop disk fills** (~ENOSPC caused a hard crash mid-session). Keep >15GB free.
- **Sonnet agents passive-die** if told to "wait"; always give bounded foreground poll orders.

## 9. EVIDENCE INDEX

- `runs/regroup_20260721/EXACT_PLAN.md` — the sprint plan (E/B/C/P/R sequences, stop-rules, P(fail))
- `runs/lanes/abc_experiment_20260721/` — E0/E1 verdicts, `e1_screen.json`, `vm_pull_v2/`
- `runs/lanes/ball_baseline_20260721/RESULTS.md` — the 0.5670 baseline
- `runs/lanes/ball_b2_seed1_20260722/RESULTS.md` — B1 timeout account + partial artifacts
- `runs/lanes/person_mixed_20260722/` — mixed-pool run, `gpu_phase_report.json`, `vm_pull/`
- `runs/lanes/person_p1_roboflow_20260721/RULING.md` — the verified PERSON negative
- `runs/manager/gpu_fleet.md` — VM/disk ledger
- `runs/handoff_20260722/ORCHESTRATOR_STATE.md` — lane-level detail (written by the orchestrator)
- Memory: `/Users/arnavchokshi/.claude/projects/-Users-arnavchokshi-Desktop-pickleball/memory/`
