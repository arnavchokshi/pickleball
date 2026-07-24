# B1 resume (WS1.1) — STATUS, 2026-07-24 (launched ~07:12Z; updated ~09:05Z)

**State: LAUNCHED, RUNNING DETACHED, PROGRESS VERIFIED.** pb.vision ball-SST build (B1) is
executing on a fresh T4 SPOT VM under the repaired, sha-pinned builder. The coordinator's 08:50Z
"stalled" alarm was investigated and is a FALSE ALARM — see the 09:05Z addendum at the bottom
(root cause: stale attempt-1 error line in an append-mode log + block-buffered stdout; the builder
never stopped). This file records the guard findings, what was launched, verifications, cost,
monitoring, and the rail-kill contingency.

## Guard findings (checked first, in order)

1. **Coordination files** (`runs/manager/inflight_lanes.md`, `runs/manager/gpu_fleet.md`, read-only):
   - `ball_b1_race_repair_20260723` row says RUNNING pid 76270, but the pid is DEAD on this Mac
     (verified `ps -p 76270` → no such process), the lane was fenced to code repair only, and its
     work is COMMITTED+PUSHED as `f07929bb8` (builder pin `4bc196f0…`, deciding review ACCEPT x4).
     No live VM job owned by it → NOT blocking, per dispatch rules.
   - `DO_NOT_DISPATCH_B1_GPU_RESUME` was **LIFTED 2026-07-23 by the integration manager** with
     recorded rationale ("B1 resume (data build) … AUTHORIZED"). Gate 1.0 (clean judge) declared
     2026-07-23 authorizes ball-lane GPU work under the pre-approved envelope. B2 remains NOT armed
     (not this task).
   - No other lane holds a ball GPU dispatch.
2. **GCP state** (project `gifted-electron-498923-h1`, auth LIVE as hello@swayformations.com):
   - `pickleball-gpu-ball-f` (VM) and `pickleball-gpu-ball-disk-f` (disk) are **GONE** — the
     owner check-in was right, the B1 media/SST disk state was torn down. The 5/7 on-disk WASB
     dependency artifacts from 2026-07-22 no longer exist in the cloud.
   - **`pickleball-gpu-ball-snap-20260722` (200GB snapshot, READY) survived** — it holds the
     pre-build state: repo clone (at old pin), venv (torch 2.9.1+cu129), WASB-SBDT @ `923462ca…`,
     WASB checkpoint, and all 7 staged train videos at `/home/arnavchokshi/pbv_media_root/<id>/max.mp4`.
   - Only other running compute: `pickleball-gpu-court23` (owner's Codex court VM) — not mine, untouched.
   - Cache infra (`pickleball-cache-*`) READY but flagged not-training-authorized; not used.

## What was launched

- **VM**: `pickleball-gpu-ball-t4`, zone `us-central1-a` (created first attempt, 2026-07-24T07:02Z),
  `n1-standard-8` + 1x **T4, SPOT**, `--instance-termination-action=STOP`,
  labels `fable-fleet=pickleball,fable-lane=ball_b1_resume_20260723,owner=arnavchokshi`.
  Boot disk `pickleball-gpu-ball-t4` (200GB pd-balanced, **auto-delete=no**) restored from
  `pickleball-gpu-ball-snap-20260722`. T4 chosen per the decode-bound evidence (identical pace to
  L4/A100 on this workload at ~1/5 the $/hr).
- **Rail**: boot-armed via startup-script metadata
  (`runs/ball_lane_20260723/b1_resume/scripts/lane_vm_startup_railed_b1resume.sh` in this worktree):
  `shutdown -P +480` → **hard poweroff 2026-07-24T15:03:56Z**; idle watchdog + preemption watcher +
  CUDA compute mode DEFAULT. NOTE: the idle watchdog cannot fire while the monitors run (its pattern
  matches the monitor cmdlines) — the 8h rail is the real cost bound.
- **Repo state on VM**: `/Users/arnavchokshi/Desktop/pickleball` (the disk mirrors the Mac absolute
  path), fetched + checked out **`f07929bb86ecb40c146ef4fbf220a7a555dbe576`** (detached).
- **Command** (launched 2026-07-24T07:12Z, nohup/setsid, real builder pid **4299**):

```
/Users/arnavchokshi/Desktop/pickleball/.venv/bin/python \
  /Users/arnavchokshi/Desktop/pickleball/scripts/racketsport/build_pbvision_ball_sst.py \
  --gallery-root /Users/arnavchokshi/Desktop/pickleball/data/pbvision_gallery_20260719 \
  --media-root /home/arnavchokshi/pbv_media_root \
  --split-manifest /Users/arnavchokshi/Desktop/pickleball/runs/lanes/pbv_pickleball_corpus_20260720/manifest.json \
  --wasb-checkpoint /Users/arnavchokshi/Desktop/pickleball/models/checkpoints/wasb/wasb_tennis_best.pth.tar \
  --wasb-repo /Users/arnavchokshi/Desktop/pickleball/third_party/WASB-SBDT \
  --teacher-confidence-min 0.90 --agreement-radius-px 20 \
  --pseudo-weight 0.25 --device cuda --resume-dependencies \
  --out /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ball_data_regroup_20260722/pbv_ball_sst.json
```

- **Log**: `/Users/arnavchokshi/Desktop/pickleball/runs/lanes/ball_data_regroup_20260722/b1_resume_20260723.log`
  (on the VM; stdout is block-buffered — a quiet log does NOT mean a dead builder; the first ~200
  bytes are a stale error line from launch attempt 1, see Findings).
- **Heartbeat (v2, hardened 09:03Z)**: `/var/tmp/b1_hb.txt` on the VM (writer:
  `/var/tmp/b1_hb2.sh`), rewritten every 60s: UTC time; builder RUNNING/DEAD + pid (precise
  pattern `[.]venv/bin/python.*build_pbvision_ball_sst`) **plus `cpu_jiffies` and
  `cpu_delta_last_cycle`** — a dead or hung inner process shows DEAD or delta≈0, so RUNNING can no
  longer mask a stall; `new_log_bytes` (total minus the 200-byte stale attempt-1 error, labeled
  IGNORE) + last NEW log line only; `OUT_EXISTS bytes/mtime` or `OUT_NOT_YET`; `dep_dirs=N/7`;
  `PREEMPTED=TRUE` if spot-reclaimed.
  (Boot log: `/var/tmp/b1resume_boot.log`; rail proof `/var/tmp/b1resume_boot_rail_armed.txt`.)

## Verifications performed before launch (all PASS)

- Builder file sha256 on VM at pin = `4bc196f0e199799c8796aa006aad42a78ae703e1efa284ed3a85adcca2058508` (matches review pin).
- Split manifest sha256 = `cf8f2518…95451b` (matches `FROZEN_SPLIT_SHA256` code constant).
- WASB-SBDT repo commit on VM = `923462cacdeb3353b84ddebdedb3f4b7a8553b0f` (matches review_r3).
- **Media: all 7 train videos re-hashed on the restored disk and verified against the builder's
  pinned `EXPECTED_SOURCE_VIDEO_SHA256` code constants — 7/7 OK.**
- Staged trees pushed from Mac (snapshot predated their staging): `data/pbvision_gallery_20260719/`
  (gallery artifacts incl. the code-pinned cv_export/api_get_metadata/video_provenance),
  `runs/lanes/pbv_pickleball_corpus_20260720/` (split manifest), and the 5 pulled per-video WASB
  dependency dirs → `runs/lanes/ball_data_regroup_20260722/pbv_ball_sst_dependencies/` —
  **116/116 files two-sided sha256 verified**.
- Compare-only exclusion: `COMPARE_ONLY_IDS` (83gyqyc10y8f, iottnc0h3ekn, o4dee9dn0ccr) are code
  constants inside `ALL_NONTRAIN_IDS`; builder iterates `TRAIN_IDS` only (the frozen 7). The three
  compare-only videos were NOT staged into the media root used by the run.
- ffmpeg/ffprobe installed (`apt-get install -y ffmpeg`) — the known "ffprobe required to SHA-bind
  PTS" clean-fail from 2026-07-22 is pre-empted.
- Reuse path CONFIRMED ENGAGED: py-spy stack shows the builder reproducing tracks from a pushed
  CSV (`wasb_csv_to_ball_track` ← builder:671) with the GPU never having spun up — i.e., no
  redundant re-inference for the reused videos. Expect `reused x4, fresh x3`
  (td2szayjwtrj's 2026-07-22 metadata lacks builder_bindings, so it recomputes; per
  `runs/lanes/ball_b1_gpu_resume_20260722/STAGED_COMMANDS.md`).

## Findings (for the builder owner)

1. **Relative-path invocation is impossible with the repaired builder.**
   `_source_file()` (builder:3555-3562) raises "must not be a symlink/alias" whenever
   `path.resolve(strict=True) != path`, which is ALWAYS true for relative paths (resolve()
   absolutizes). The exact EXACT_PLAN/RESULTS.md command (relative paths, run from repo root)
   fail-closes in seconds. Launch attempt 1 died on this; attempt 2 with fully absolute
   `--gallery-root/--media-root/--split-manifest/--wasb-checkpoint/--wasb-repo/--out` runs fine.
   The repair-lane tests presumably used absolute tmp_path fixtures and never saw it.
2. **Known perf defect governs the wall** (`runs/lanes/ball_b1_race_repair4_20260723/PERF_BUG_io_decode.md`):
   unconditional per-frame byte-identity reproduction for ALL 7 videos (~155,515 frames @ ~10.4fps
   ≈ ~4h CPU-only, GPU idle) on top of fresh inference for 3 videos (~69,443 frames @ 283.9-494.1
   fr/min ≈ 2.3-4.1h). Slow, not wrong. It also makes ANY rerun pay ~4h even with full reuse.
3. Monitoring gotcha fixed mid-lane: heartbeat/watchdog loops whose cmdline contains the builder
   name self-match `pgrep -f`. The corrected heartbeat uses a bracketed pattern. (Also: `pkill -f`
   of such a pattern kills your own SSH session — cost two rc-255 mysteries.)

## Sizing, rail risk, and contingency (READ THIS IF THE RAIL FIRES)

- Estimated total wall from 07:12Z: **5.9-8.5h** (reused-video reproduction ≈2.3h + fresh
  inference+reproduction ≈3.3-5.9h + snapshot/overheads ≈0.3h) → completion ~**13:05-15:40Z**.
  **REVISED 09:05Z with measured rates**: reproduction is running at 15.7-19.5 fps on this CPU
  (better than the 10.4 fps planning number; py-spy frame-counter samples 08:52-09:01Z) →
  ETA ~**12:15-15:30Z** depending on which video the counter is currently in (per-frame cost
  scales with each video's frame-time table, so per-video rates vary). Most of the distribution
  completes under the rail; only the extreme slow tail risks a rail-kill ≤~30 min short.
- The rail fires **15:03:56Z** (7.86h of runway). Mid-band completes under it; the slow tail may be
  **rail-killed near the end (~40min short, worst case)**. I attempted to extend the rail to +585
  and the permission system denied canceling the armed shutdown — respected, not worked around.
  Owner can extend anytime with:
  `gcloud compute ssh pickleball-gpu-ball-t4 --zone=us-central1-a -- 'sudo shutdown -c && sudo shutdown -P +180 "B1 rail extension"'`
- If the rail (or a spot preemption) kills the run: the disk (`auto-delete=no`) keeps every
  completed per-video dependency dir. Restart the VM (`gcloud compute instances start
  pickleball-gpu-ball-t4 --zone=us-central1-a` — the startup script re-arms an 8h rail at boot),
  then relaunch the SAME command above (nohup/setsid). `--resume-dependencies` will reuse every
  video whose artifacts completed; only the interrupted video re-infers (plus the unavoidable ~4h
  reproduction pass, defect #2). Est. resume cost ≤ ~$2.
- **Teardown is NOT this task's job**: after the gate verdict is pulled and ruled, the manager
  decides disk deletion (same ruling discipline as 2026-07-22). Do not delete
  `pickleball-gpu-ball-snap-20260722` — it is the only surviving media/env restore point.

## Acceptance gate (unchanged, frozen)

`accepted_windows >= 1,000` across `>= 5` of 7 train sources; zero permanent-holdout rows; every
window decodes; each row carries agreement reason + exact dependency hashes (conf ≥ 0.90, radius
20px, or the preregistered temporal bridge). Otherwise the named negative
`PBV_BALL_INSUFFICIENT_AGREEMENT` — either verdict with complete evidence is a valid lane outcome.
No training, no threshold changes in this lane. SST corpus registration happens in a LATER task
(ledger held by another agent).

## Cost

| item | value |
|---|---|
| instance | `pickleball-gpu-ball-t4` n1-standard-8 + 1x T4 SPOT, us-central1-a |
| $/hr band | ~$0.20-0.40 (T4 spot band per fleet history) + pd-balanced 200GB (~$0.03/h) |
| hard bound | 8h rail → **max ~$1.9-3.5 this boot** (cap $10; ledger row est 4-8h / $1.5-3.5) |
| ledger row | committed `62341a2` in `runs/ball_lane_20260723/owner_packet/GPU_COST_LEDGER.md` (BEFORE launch) |

## How to check progress (from the Mac)

```
# one-shot: heartbeat + builder liveness + tail
gcloud compute ssh pickleball-gpu-ball-t4 --zone=us-central1-a \
  --command="cat /var/tmp/b1_hb.txt; tail -5 /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ball_data_regroup_20260722/b1_resume_20260723.log"

# per-video dependency completion (grows 5 -> 7 as fresh videos finish)
gcloud compute ssh pickleball-gpu-ball-t4 --zone=us-central1-a \
  --command="ls /Users/arnavchokshi/Desktop/pickleball/runs/lanes/ball_data_regroup_20260722/pbv_ball_sst_dependencies/"

# done when: OUT_EXISTS appears in /var/tmp/b1_hb.txt, i.e. pbv_ball_sst.json written
# then: pull pbv_ball_sst.json + log + the 3 fresh dependency dirs, two-sided sha256,
# read the gate verdict from the manifest summary, and STOP the VM. (Later task.)
```

VM status / rail check: `gcloud compute instances describe pickleball-gpu-ball-t4 --zone=us-central1-a --format='value(status)'`

## ADDENDUM 2026-07-24 ~09:05Z — coordinator stall alarm: investigated, FALSE ALARM

The 08:50Z heartbeat (`builder=RUNNING`, `log_bytes=200`, `log_last=` attempt-1's
symlink/alias error) was read as "wrapper running, inner builder errored out." Reconstruction
with fresh evidence says otherwise — the builder never stopped:

- **pid 4299 IS the inner builder, not a wrapper** (py-spy attaches to it and dumps
  `build_pbvision_ball_sst.py` frames; its cmdline is the absolute-path invocation with
  `--resume-dependencies`; `setsid` did not fork).
- **The log line is attempt 1's corpse, not attempt 2's state**: log mtime is frozen at
  **07:10:19Z** — before pid 4299 existed (07:12Z). The error references snapshot dir
  `…_wv4gt8th` (attempt 1's, deleted before relaunch); the live run's snapshot dir is
  `…_pq2upnff`. Attempt 2 has written 0 log bytes because its stdout is **block-buffered**
  and the builder is in a multi-hour phase with no flush — a quiet log here is normal.
- **Liveness + forward progress proven, not asserted**: cumulative CPU 1h32m at 08:52Z
  (92.4% duty since launch); py-spy loop-counter samples `row.frame` = 3510 (08:52:14Z) →
  3948 → 5677 (08:56:10Z) → 6399 (08:56:47Z) → **11568 (09:01:15Z)** — monotonic,
  ~15.7-19.5 fps through the known `PERF_BUG_io_decode.md` reproduction phase (GPU
  correctly idle during it). No invocation fix or relaunch was needed; killing the run
  would have wasted ~1.75h of valid compute.
- **Monitoring hardened (heartbeat v2)** so this class of false alarm cannot recur:
  `cpu_delta_last_cycle` per 60s cycle (dead/hung ⇒ DEAD or ≈0; verified live showing
  ≈100% of one core), stale attempt-1 bytes excluded and labeled IGNORE, `OUT` size/mtime,
  `dep_dirs=N/7`. Watch `dep_dirs` go 5→7 and GPU util rise when fresh-video inference starts.
- Incidental ops lesson (cost three rc-255 SSH deaths): `pkill -f <pattern>` run over SSH
  kills the SSH session itself whenever the pattern appears literally anywhere in the sent
  command string (heredoc bodies included). Control such loops via a small script file
  invoked by a clean path (`/var/tmp/hbctl.sh`), with bracketed patterns.
- **Rail decision unchanged**: rail fires 15:03:56Z; measured-rate ETA 12:15-15:30Z. Not
  re-attempting the denied shutdown-rail change. If the slow tail clips the rail, the
  contingency above applies (restart VM, relaunch same command, `--resume-dependencies`
  reuses everything completed; est ≤~$2).
- Cost estimates unchanged (row `62341a2` stands): the VM was never idle — the builder has
  held ~100% of a core continuously since 07:12Z; bound remains the 8h rail (~$1.9-3.5).
