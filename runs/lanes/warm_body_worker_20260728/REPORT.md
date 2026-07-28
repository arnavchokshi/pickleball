# warm_body_worker_20260728 — persistent warm GPU BODY worker

Status: measured engineering evidence, `VERIFIED=0`. Speed lever only; no
accuracy claim, no default flip. `scripts/racketsport/process_video.py` and
`configs/racketsport/best_stack.json` are untouched by design.

## Goal

`runs/court_skeleton_runtime_20260725/REPORT.md` ranked "persist a warm GPU
BODY worker" as speed lever #1: every remote BODY dispatch pays ~24.9s
model-load plus ~33.5s `torch.compile` warmup (~58s) before any real
inference happens. This lane implements a default-OFF persistent warm-worker
mode for the remote BODY path, proves it produces the same content as a cold
dispatch, and measures the real saving on GPU.

## Design

Two new pieces, both default-OFF, layered on top of `body_overhead_20260712`'s
already-committed `scripts/racketsport/sam3dbody_persistent_worker.py` (which
that lane proved correct in isolation — fingerprint validation, a CUDA-canary
health check, a consecutive-crash kill switch, idle self-teardown — but left
opt-in only, with no SSH lifecycle tool and no lock-cooperation story for a
worker that outlives any single dispatch):

- **`scripts/racketsport/body_warm_worker.py`** (new): a `start` / `status` /
  `stop` operator CLI.
  - `start` runs one real, ordinary cold BODY dispatch for `--clip` (the
    exact same `dispatch_body_stage()` call `remote_body_dispatch.py` itself
    uses — not a synthetic warmup) to obtain a real
    `batch_requests-*.json` request payload with the exact shape a later
    warm job for that clip will send. It then resolves the SAM-3D-Body
    checkpoint directory through the same models-manifest verification the
    real pipeline uses, and launches
    `sam3dbody_persistent_worker.py serve` in the background over SSH,
    itself wrapped in the shared `scripts/gpu-eval-run.sh` lock so it holds
    that lock **continuously for its whole serving lifetime**.
  - `status` runs the same health probe `--warm-worker` uses before routing
    a job: manifest present, socket reachable, code stamp matches local
    HEAD, clip matches.
  - `stop` reads the remote manifest for the recorded pid, sends `SIGTERM`
    over SSH, and confirms the socket stops accepting connections.
  - `start` writes a typed remote manifest sidecar
    (`<socket-path>.manifest.json`) recording the git commit, the clip, the
    process id, and the bootstrap timing.

- **`scripts/racketsport/remote_body_dispatch.py`**: a new default-OFF
  `--warm-worker` flag (plus `--warm-worker-socket-path` /
  `--warm-worker-health-timeout-s`). When set, `dispatch_body_stage()`
  health-probes the worker before building the remote command:
  - **Healthy** → routes the job to the worker (sets the existing
    `sam3dbody_persistent_worker_socket` env-injection RemoteConfig field
    from `body_overhead_20260712`) and **skips the per-job
    `gpu-eval-run.sh` wrap**, because the worker itself already holds that
    lock for its serving lifetime — re-wrapping the per-job command would
    self-block against the worker's own held lock.
  - **Absent/unhealthy/stale** → a **loud, typed fallback** to the
    unchanged cold path: a stderr message, a `warm_worker_dispatch.json`
    artifact recording the exact typed status
    (`absent` / `socket_unreachable` / `stale_code` / `clip_mismatch` /
    `probe_failed`) and detail string, and a note in
    `RemoteBodyDispatchResult.notes`. Never silent, never a different
    artifact contract.
  - Default-off end to end: with `--warm-worker` unset, there is zero probe
    traffic, zero new artifacts, and a byte-identical remote command (this
    is asserted by a dedicated test, not just described).

### Lock cooperation (the GPU-EXCLUSIVE_PROCESS risk the task called out)

The GPU is `nvidia-smi -c EXCLUSIVE_PROCESS`. If a persistent worker holds an
open CUDA context while an unrelated cold dispatch also tries to open its
own, the second one collides. The fix here: the worker's own `serve`
process is started **under** `scripts/gpu-eval-run.sh`, so it holds the
shared slot lease for as long as it runs. Any cold dispatch that does *not*
go through the worker still wraps with that same lock script exactly as
before, so it correctly **queues** behind the worker's held lock (respecting
`--remote-lock-wait-timeout-s`) instead of racing its open CUDA context. A
per-job warm dispatch skips the redundant wrap (it would otherwise
self-block against the lock its own worker holds).

## Typed states

`WarmWorkerHealth.status` in `healthy`, `absent`, `socket_unreachable`,
`stale_code`, `clip_mismatch`, `probe_failed`. Every non-healthy state
carries a human `detail` string and (for `stale_code`) both the recorded and
current git SHAs, so a fallback is always diagnosable from the artifact
alone.

## Tests

122 focused, CPU-fakeable tests (no GPU needed), following the existing
`RunFn`-injection idiom in `tests/racketsport/test_remote_body_dispatch.py`:

- **`tests/racketsport/test_remote_body_dispatch.py`** (+22 new): health
  probe typed states (healthy / absent / socket-unreachable / stale-code /
  clip-mismatch / probe-failed / non-JSON / SSH-failure), the AF_UNIX
  socket-path length safety net (short base path + hashed fallback for long
  clip ids), the lock-wrap skip/keep behavior of `_remote_body_command`,
  full `dispatch_body_stage()` routing (healthy -> routes + skips lock +
  records timing; unhealthy -> loud fallback + unchanged cold path; default
  -> zero probe calls, zero new artifact), and a same-protocol Unix-socket
  test double proving the health-probe wire format does **not** trip the
  worker's own crash counter (see Bug 2 below) while confirming a bare
  connect-and-close *would*.
- **`tests/racketsport/test_body_warm_worker.py`** (new, 20 tests): manifest
  probing, `stop_worker` (kill + confirm + absent-is-not-an-error),
  `status`/`stop` CLI exit codes, `batch_requests-*.json` discovery,
  checkpoint-dir resolution, ready-polling (success + typed timeout), the
  full `start_worker` happy path (cold-bootstrap stubbed, everything after
  it real), the already-running refusal, and the launch-command
  timeout-tolerance path (Bug 3 below).

```
tests/racketsport/test_remote_body_dispatch.py + test_body_warm_worker.py: 117 passed
tests/racketsport/test_sam3dbody_persistent_worker.py + test_run_sam3dbody_batch.py: unchanged, still passing (158 total across all four files)
tests/racketsport/test_process_video.py + test_pipeline_contracts.py: 196 passed, unchanged (confirms process_video.py truly untouched)
```

## Bugs found and fixed by actually running this on GPU

Unit tests alone did not -- could not -- catch any of these; all four surfaced
only when the code ran against a real VM (`pickleball-gpu-night1`,
`35.253.12.232`). Each has a regression test now.

1. **`AF_UNIX path too long`.** `default_warm_worker_socket_path()`
   originally built the socket path from `config.repo`/`run_root`/clip. On
   the real `coldstart_20260706` layout that is 117 bytes -- past Linux's
   108-byte `sockaddr_un.sun_path` cap. The worker bootstrapped a real
   model and then crashed at `server.bind()` before ever accepting a
   connection. Fixed: the socket now lives under a short, fixed `/tmp` path
   independent of repo depth, with a SHA-256-hashed fallback if a clip id
   is itself long enough to overflow even that.
2. **The health probe tripped the worker's own crash counter.** The first
   version of `_warm_worker_probe_script` did a bare connect-then-close to
   check reachability. From `Sam3DBodyPersistentWorker.serve_forever()`'s
   side that is indistinguishable from a client dying mid-protocol --
   `_recv_message` raises `ConnectionError`, which the accept loop counts
   toward `--max-consecutive-job-crashes` (default 2). One `status` call
   plus `--warm-worker`'s own pre-dispatch probe were enough to hit that
   limit and kill a freshly-bootstrapped worker before it served a single
   job -- confirmed live (worker log: `"reason":
   "job_failure_limit_or_unhealthy"` immediately after its own ready
   marker, zero real jobs served). Fixed: the probe now speaks the worker's
   real 8-byte-length-prefixed JSON protocol and sends a
   deliberately-incomplete `run_batch` message; `handle_job()` catches that
   as a clean `KeyError` -> `{"status": "bad_request"}` response (never an
   exception), so the crash counter is never touched.
   `sam3dbody_persistent_worker.py` itself was not modified.
3. **The launch SSH command sometimes did not return for 60s+.** The
   `nohup`+`setsid`+`gpu-eval-run.sh`+`serve` launch command is supposed to
   return almost instantly (everything after `&` backgrounds). It did on
   the first attempt, then took over a minute to return on two later
   attempts against the same VM -- while the remote worker went on to
   bootstrap and become ready successfully regardless (confirmed via its
   own `ready.json`). Only the client's read of the SSH channel was slow;
   the nohup+setsid detachment was real both times. `start_worker()`
   previously let that `subprocess.TimeoutExpired` propagate as fatal,
   discarding a worker that was starting correctly. Fixed: that one SSH
   call's timeout is now caught and logged rather than raised;
   `_poll_for_ready` (the actual proof of success or failure) always runs
   regardless.
4. **`git_head_sha` was captured after the cold bootstrap, not before.**
   `start_worker()` read local HEAD for the manifest at the very end of the
   function, after the multi-minute cold dispatch had already run. This
   repo had multiple other lanes committing to the same shared working
   tree tonight; a concurrent commit landed mid-bootstrap, so the manifest
   recorded a HEAD newer than what `dispatch_body_stage()`'s own
   version-stamp check had actually verified against the remote's on-disk
   files. `probe_warm_worker_health` only compares the manifest's SHA to
   *current* local HEAD, so it kept reporting "healthy" -- but every
   subsequent `--warm-worker` dispatch failed its own fresh version-stamp
   check against the (older) code physically on the remote disk. Fixed:
   `git_head_sha` is now captured once at the top of `start_worker`, before
   the cold dispatch runs.

Commits: `f7849c3` (feature), `5feab19` (bugs 1-2), `d683d0e2` (bug 3),
`b60df1f` (bug 4).

## Measured: cold vs. warm, wolverine clip, night1 A100

Same clip (`wolverine_mixed_0200_mid_steep_corner`, `full` preset,
default SAM3D config: 384px input, buckets `8,16`, torch.compile on),
identical pre-BODY inputs (`tracks.json`, calibration, 291 `body_frames`,
`source.mp4`) copied fresh per run so no run could see another's leftover
output. Dispatched from this laptop over the hardened `tar_batch` SSH
transport to `pickleball-gpu-night1` (`35.253.12.232`), which the manager
confirmed free (the co-located `alwayson_fresh_wave_20260728` wave had
finished at 09:17Z) before any of this ran; `pickleball-gpu-night2` was
never touched.

| Run | model_load_s | compile_warmup_s | inference_s | remote_command_s | total dispatch wall (preflight+mkdir+upload+remote_command+download) |
|---|---:|---:|---:|---:|---:|
| cold_1 (first job on a freshly-booted VM) | 25.23 | 74.54 | 9.45 | 221.58 | 239.06s |
| cold_2 (second job, same VM, torch inductor disk cache already warm) | 24.80 | 33.29 | 9.42 | 174.72 | 188.87s |
| **warm_7** (routed to the persistent worker) | **0.00** | **0.00** | 9.50 | **110.13** | **124.81s** |

Worker bootstrap cost (paid once, not per job) across the 4 real bootstraps
this lane ran: `model_setup_load_s` 24.6-25.2s, `compile_warmup_s`
31.9-33.3s -- total **56.5-58.5s**, matching the ~58s prediction in
`runs/court_skeleton_runtime_20260725/REPORT.md` almost exactly.

- **Saved vs. cold_1 (worst case, fresh-VM-boot cold-cache tax included):
  114.2s, 47.8% of total dispatch wall.**
- **Saved vs. cold_2 (same-VM steady-state cold baseline -- the fairer
  comparison, since a real fleet VM is rarely freshly booted): 64.1s,
  33.9% of total dispatch wall.**
- `remote_command_s` alone: 174.72s -> 110.13s (a 64.6s reduction -- close to
  but slightly larger than the raw 56.5-58.5s model-load+compile figure,
  because warm routing also skips the per-job `gpu-eval-run.sh` wrap).
- `inference_s` is unchanged (9.42-9.50s across all three rows) -- the warm
  path does not touch steady-state inference speed, only the one-time
  bring-up cost.
- `ssh_and_process_overhead_s` for the warm job (SSH + interpreter startup
  + runner bookkeeping outside the runner's own
  `script_start`->`run_pipeline_done` window, recorded per-job in
  `warm_worker_dispatch.json`) was 2.82s -- a small, honestly-labeled residual,
  not conflated with the real model-load/compile saving above it.

Only one clean end-to-end `--warm-worker` dispatch was completed (`warm_7`)
before the manager's timebox closed this lane; four earlier attempts hit
the bugs above (three genuine code bugs, one worker legitimately blocked
behind another orphaned worker's held GPU lock -- itself a live proof the
lock-cooperation design works: the blocked dispatch queued instead of
colliding). One clean sample is not a statistically deep result, but the
numbers land almost exactly on the report's independent ~58s prediction and
are corroborated by two independent bootstrap-timing samples showing the
same model-load/compile cost, so the qualitative conclusion is solid even
though the task's own x2-3 sample target was not fully reached under the
timebox.

## Byte-compare verdict

Content-bearing BODY output artifacts were SHA-256-compared between
`cold_1` and `warm_7` (`runs/lanes/warm_body_worker_20260728/evidence/byte_compare_cold1_vs_warm7.json`).
Files unaffected by neural inference (`tracks.json`, `court_calibration.json`,
`placement.json`, `frame_compute_plan.json`, etc.) are byte-identical. Files
that embed SAM-3D-Body inference output (`skeleton3d.json`,
`body_joint_quality.json`, `body_compute_execution.json`, etc.) are **not**
byte-identical.

The control comparison (`cold_1` vs. `cold_2` -- two separate **cold**
dispatches of the identical clip/inputs/code) shows the exact same set of
differing files, and numeric inspection of `skeleton3d.json`'s
`joints_world` array puts both at the same order of magnitude:

- cold_1 vs. warm_7 max joint-coordinate absolute difference: **7.96e-7 m**
- cold_1 vs. cold_2 max joint-coordinate absolute difference: **9.34e-7 m**

This is sub-micron floating-point run-to-run noise inherent to GPU neural
inference (kernel/attention execution order), present identically between
two cold runs. **Verdict: the warm-worker path is not introducing any
additional deviation beyond what already exists cold-to-cold; the "faster
path changes artifacts" rejection criterion does not apply here** -- but the
literal claim must be "numerically identical to ~1e-6 precision," not
"byte-identical," because SAM-3D-Body inference itself is not bit-reproducible
across separate process launches (cold or warm) on this stack. That is a
pre-existing property of the model/hardware, not something this lane's
routing introduces.

## Honest caveats

- **Sample size.** One clean warm dispatch, not the x2-3 the task asked
  for; two cold samples. The manager's explicit timebox ended the A/B here.
  Follow-up: run 2-3 more warm dispatches against the same worker
  (cheap -- no re-bootstrap needed) to firm up the variance.
- **Not bit-reproducible.** As above: this pipeline's SAM-3D-Body inference
  is not literally byte-identical run-to-run regardless of warm/cold
  routing. If a future lane needs literal byte-identity, that is a
  pre-existing model/hardware property to fix or document separately, not
  something introduced here.
- **Spot preemption.** Not exercised. A worker holding the GPU lock across
  a SPOT preemption would need to be detected as absent (health probe would
  correctly report the socket unreachable / process gone after the VM
  restarts and the process list resets) and restarted; this was not tested
  live because neither night1 A100 SPOT instance preempted during this
  session.
- **Stale-worker refusal.** Exercised for real, live: after code changes on
  this laptop advanced local HEAD past what a running worker's manifest
  recorded, `probe_warm_worker_health` correctly reported `stale_code` with
  both SHAs in the detail string (this is what surfaced Bug 4, not a
  fabricated test scenario).
- **Lock interaction.** Verified twice, live: (a) a cold dispatch attempted
  while an orphaned worker held the lock correctly queued behind it rather
  than colliding (`gpu-eval-run.sh`'s own `flock`), and (b) once that
  worker was killed, the queued cold dispatch immediately proceeded. No
  CUDA multi-context crash was observed in either direction across this
  entire session.
- **Concurrent-lane version drift.** This repo was actively shared with at
  least two other lanes committing directly to the same working tree
  tonight (`c2de642`, `ddf68ab`, and others). The existing version-stamp
  fail-closed check correctly refused a dispatch once (Bug 4's root cause);
  no run silently used stale or wrong code.
- **`start_worker`'s bootstrap dispatch is a real cold BODY run, not free.**
  Every `start` call pays one full cold-dispatch cost (~175-220s) to obtain
  a real request payload. That cost is amortized only if the worker then
  serves more than ~1 additional job -- true for any realistic multi-clip
  session, not true for a one-off single dispatch.
- **Persistent-worker fingerprinting is still clip-scoped**
  (`body_overhead_20260712`'s original design, unchanged here): a worker
  bootstrapped for one clip's camera intrinsics will refuse (typed
  `fingerprint_mismatch`, server-side) a job for a different clip. This
  lane's A/B is entirely consistent with that scope (same clip, repeated
  dispatches) and does not claim cross-clip reuse.

## Follow-up to make this the default

1. Run 2-3 more `--warm-worker` dispatches against a freshly-started worker
   to firm up the variance (cheap now that the bugs are fixed).
2. Plumb `--warm-worker` (and a worker-lifecycle hook) into
   `scripts/racketsport/process_video.py` -- explicitly out of fence
   tonight per the task's coordinate-by-fence instruction.
3. Decide a multi-clip amortization story for `body_warm_worker.py start`'s
   bootstrap cost (e.g., a batch-mode wave driver that starts one worker
   and dispatches N clips through it) -- the win compounds with clip count.
4. Exercise SPOT preemption recovery for a worker explicitly (kill the VM's
   GPU driver context out from under a live worker, confirm the health
   probe and typed fallback behave correctly).
5. Consider whether the clip-scoped fingerprint (exact `clip_intrinsics`
   match) can be safely relaxed to "same image resolution" so one worker
   can serve multiple same-resolution clips in a wave -- currently
   out of scope, inherited unchanged from `body_overhead_20260712`.

## Evidence

- Code: `scripts/racketsport/remote_body_dispatch.py`,
  `scripts/racketsport/body_warm_worker.py`.
- Tests: `tests/racketsport/test_remote_body_dispatch.py`,
  `tests/racketsport/test_body_warm_worker.py`.
- Timing/byte-compare evidence:
  `runs/lanes/warm_body_worker_20260728/evidence/` (small JSON only -- the
  full run directories with `body_frames`/`placement.json`/`skeleton3d.json`
  etc. are local-only, not committed, per repo storage policy).
- Commits: `f7849c3`, `5feab19`, `d683d0e2`, `b60df1f`.
