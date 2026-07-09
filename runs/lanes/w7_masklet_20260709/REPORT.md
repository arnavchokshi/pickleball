# LANE w7_masklet_20260709 — P2-4 SAM-Body4D masklet-conditioning spike

## VERDICT: NO-ATTEMPT (candidate arm) + baseline BANKED

No kill criterion fired on the candidate's merits. The candidate arm was hard-blocked by the
session's permission system (auto-mode classifier), which denied executing/installing the external
`github.com/gaomingqi/sam-body4d` code (3 denials: the detectron2 `git+` build dep, `pip install -e
models/sam3`, and `pip install -e .` of sam-body4d itself — all classified "Code from External /
Untrusted Code Integration"). Per protocol step 1, a NO-ATTEMPT with the missing prereq named beats
a fabricated bench. **Missing prereq: owner/manager-level permission grant (Bash allowlist rule or a
non-auto-mode session) to execute the vetted third-party sam-body4d repo on a fleet VM.**

The VM was torn down early rather than burning the K1 60-min window against an absolute blocker.

## EVIDENCE BASE (Mac-side, read before any VM)
- RULINGS.md R5b + sota_body_synthesis.md: SAM-Body4D (arXiv:2512.08406, MIT, repo
  gaomingqi/sam-body4d) = training-free masklet-prompt conditioning on the same SAM-3D-Body/MHR
  base; claimed ~2x batching win, no reported accuracy regression; Fig. 8 per-stage ms tables were
  UNEXTRACTABLE in the research pass (qualitative claims only banked).
- NORTH_STAR P2-4 (line 1161): concrete recipe (repo URL + conditioning mechanism) -> attempt was
  justified; gate >=20% raw-noise reduction or runtime win, kill >90s/clip.
- Prior-bench protocol reused from w5_fastbody_bench_20260708 (footguns honored: --allow-overwrite,
  own clip-dir copy — never another lane's, fail-loud runtime summary).

## CANDIDATE ARM — how far setup got before the wall
| step | result |
|---|---|
| repo clone (gaomingqi/sam-body4d) | OK — HEAD `21af1020979e` (2026-01-27 "update uni-cam-int"), MIT |
| miniconda install | OK |
| conda env `body4d` python=3.12 | OK (required conda ToS accept — new-VM gotcha) |
| torch 2.7.1+cu118 / torchvision 0.22.1 / torchaudio 2.7.1 | OK, installed |
| detectron2 `git+...@a1ce2f9` install | DENIED by permission classifier |
| `pip install -e models/sam3` | DENIED |
| `pip install -e .` (sam-body4d) | DENIED |

Additional prereq risks discovered for any re-attempt (would have been next even with permissions):
1. SAM-3 (`facebook/sam3`) and SAM-3D-Body (`facebook/sam-3d-body-dinov3`) checkpoints are
   HF-GATED (prior access approval required). VM HF token exists (37 bytes, from cold start; it
   downloaded sam-3d-body-dinov3 before) but sam3 access approval is UNVERIFIED.
2. Full pipeline needs FIVE model families: SAM-3, SAM-3D-Body, MoGe-2, Diffusion-VAS (x2 ckpts),
   Depth-Anything-V2-Large — a diffusion-based occlusion module rides in the loop. This is heavier
   than the "cheap spike" framing; a realistic re-attempt should budget checkpoint download +
   K1-clock accordingly, and consider whether a masklet-prompts-only subset (SAM-3 masklets ->
   our existing mask_prompt_mode=manifest input, skipping Diffusion-VAS) is the actual cheap A/B.
3. Their entrypoint is video-in/meshes-out (`scripts/offline_app.py --input_video`); wiring output
   joints into our matched-frame divergence metric needs an adapter (their MHR outputs vs our
   `sam3d_body_joints` — same family, so joint mapping is tractable but not free).

## BASELINE ARM — BANKED (full promoted stack defaults, H100, non-degraded)
Clip: wolverine_mixed_0200_mid_steep_corner (internal-val). Dispatch via
scripts/racketsport/remote_body_dispatch.py, tar_batch transport, status "ran".

| metric | value |
|---|---|
| remote_command wall (BODY stage on VM) | **307.179 s** |
| end-to-end dispatch wall (incl transport) | 420.449 s |
| attributed_s (instrumented BODY phases) | 267.64 s |
| inference steady (672 frames @ bucket 16) | 5.017 s |
| ms_per_person_steady | 7.49 ms |
| compile warmup | 45.31 s (37.9 s of it the size-8 bucket that got only 8 real frames — known waste, flagged in w4) |
| person_frame_count | 680 |
| GPU | H100-80GB (a3-highgpu-1g SPOT, ase1-c) |
| degraded mode? | NO — batched tier2, torch.compile, mask_prompt manifest mode |
| version stamp | verified=true, sha 87a852f5c, 76 files checked, 0 drifted |

Cross-check: w5_fastbody baseline on the same clip/same SKU was remote_command=316.052 s — this
run is consistent (2.8% faster), so the baseline is reproducible and trustworthy as the comparison
anchor for any future candidate arm.

Candidate-vs-baseline table, GPU-util trace, joints-divergence distribution: NOT PRODUCED (no
candidate ran). No accuracy claims made.

## VERSION STAMPS
- Repo (Mac + VM): committed HEAD `87a852f5c10a217b053bcda8c2bbb1b79466c621`, synced via
  --sync-remote-code (git bundle), remote md5 verification 76/76 files 0 drift
  (artifacts/baseline_h100/remote_version_verification.json).
- sam-body4d candidate: `21af1020979ef32ddf6be3597ef59a68bad2f1bf` (cloned, never executed).

## VM LIFECYCLE + COST (honest span)
- CREATE issued 2026-07-09T08:19:45Z; RUNNING + first-try SSH 08:23:06Z (a3-highgpu-1g H100 SPOT,
  asia-southeast1-c, pd-balanced 200GB FROM pickleball-fleet-snap-20260708-w6close, SPOT+STOP,
  labels fable-lane=w7-masklet,fable-fleet=pickleball).
- DELETE 08:48:36Z -> confirmed 08:49:07Z; fleet list-confirmed after delete: only
  pickleball-h100-w7ball (RUNNING, other lane's — untouched) + pickleball-a100-fleet1 (TERMINATED,
  pre-existing).
- Uptime 0.489 h x $0.57-4.25/hr = **$0.28-2.08 (mid ~$1)**. Zero preemptions. K4 (2.5 h) never
  approached.

## HONEST ISSUES
1. **Boot hygiene deviation from brief**: the w6close snapshot's repo lives at
   `~/coldstart_20260706/repo` (not `~/pickleball`); it booted 101 commits behind origin with the 2
   by-design vendor overlay lines + a `.venv` untracked dir — reset --hard + fetch + sync per
   protocol.
2. **Snapshot is missing two REQUIRED best_stack artifacts** (both gitignored, so no git sync can
   deliver them): `runs/waveb_confidence_gate_20260702T183158Z/calibration_curves.json` and
   `models/checkpoints/court_unet_v2/court_model_v2.pt` (287 MB). best_stack manifest validation
   hard-fails at remote_body_dispatch import time on the missing paths — BODY cannot run from this
   snapshot without them. I scp'd both and sha256-verified against the manifest's pinned hashes
   (exact match). **Next snapshot cut must bake these, or every BODY lane pays this tax.** This is
   a NEW failure mode introduced by best_stack rev 6 path validation + the older snapshot.
3. Dispatch needed `--allow-dirty`: concurrent-session dirt (~90 tracked files incl.
   configs/racketsport/best_stack.json) plus the Mac HEAD advancing mid-lane (a concurrent lane
   committed 87a852f5c between my sync and dispatch) caused two dispatch aborts before the clean
   run. Known w4-class gotcha; remote md5 verification still passed 76/76 against committed blobs.
4. **Anti-passive-wait violations (mine)**: I twice ended turns waiting on background monitors
   before the coordinator's resume; corrected to bounded foreground polling. Lane lost ~10 min of
   VM idle to this (included in the honest cost span).
5. Permission denials are session-mode facts, not candidate facts: nothing here is evidence for or
   against masklet conditioning itself.
6. `configs/ssh/a100_known_hosts` was refreshed for the VM IP (standard fleet protocol) — that edit
   plus the two artifact scp's are the only things this lane touched outside its lane dir; repo
   source untouched, no best_stack/manifest changes.

## NEXT (manager rules)
1. If P2-4 is still wanted: re-dispatch with an explicit permission grant for
   `gaomingqi/sam-body4d` execution (settings allowlist or non-auto session). Budget: HF sam3
   access check BEFORE VM create, ~10-20 GB checkpoint pulls, and the 5-model env — the K1 60-min
   clock is tight for the full pipeline; consider the masklets-only A/B subset instead.
2. Fold calibration_curves.json + court_model_v2.pt into the next fleet snapshot cut.
3. Baseline 307.2 s H100 anchor is banked and reproducible for any future BODY challenger bench.
