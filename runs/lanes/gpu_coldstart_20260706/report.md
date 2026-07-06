# P0-1 GPU cold-start report — pickleball-a100-fleet1 (2026-07-06)

## Result: PASS. VM provisioned, cold-started, smoke-tested, left RUNNING + idle as fleet GPU #1.

## VM details
- name: `pickleball-a100-fleet1`
- zone: `asia-southeast1-a` (project `gifted-electron-498923-h1`) — succeeded on the FIRST zone tried
  (the mission's fallback list order asia-southeast1-a/b/c, us-central1-a/b/f, us-west1-b, europe-west4-a
  was not needed past entry 1)
- machine type: `a2-highgpu-1g` (1x NVIDIA A100-SXM4-40GB, driver 580.159.03)
- provisioning: SPOT, `--instance-termination-action=STOP`, `--maintenance-policy=TERMINATE`
- image: `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` / project `deeplearning-platform-release`
  (see "Deviation 1" below — `pytorch-latest-gpu` no longer exists)
- boot disk: 200GB requested (auto-grew from image's 100GB via cloud-init on first boot; confirmed
  194G filesystem, 158G free after full cold start + checkpoint + vendor clones)
- labels: `fable-lane=coldstart,fable-fleet=pickleball,owner=arnavchokshi`
- startup-script: `scripts/fleet/lane_vm_startup.sh` (confirmed effective: `nvidia-smi -q -d COMPUTE`
  shows `Compute Mode: Exclusive_Process` on the running VM)
- external IP: `34.143.175.207`
- created: `2026-07-06T18:55:16Z` (`2026-07-06T11:55:16.969-07:00`)
- exact SSH command: `gcloud compute ssh pickleball-a100-fleet1 --zone=asia-southeast1-a --project=gifted-electron-498923-h1`
- estimated cost: ~$1.1-1.3/hr (same zone/shape as the prior VM `pickleball-a100-spot-ase1a` per
  RESET_HANDOFF §7's own figure; well under the $5/hr owner cap)

## Preflight (before create)
- gcloud auth: confirmed ACTIVE as `hello@swayformations.com`, project `gifted-electron-498923-h1`
  (per mission — did not need owner re-login)
- Quota: `PREEMPTIBLE_NVIDIA_A100_GPUS` = 16 (0 used) in all 4 candidate regions (asia-southeast1,
  us-central1, us-west1, europe-west4) — quota was never the constraint; capacity/stockout would have
  been, but zone 1 succeeded immediately
- `a2-highgpu-1g` confirmed offered in asia-southeast1-a before create attempt
- default network + `default-allow-ssh` (0.0.0.0/0 tcp:22) firewall rule already present in this
  project — no networking setup needed
- Local HF token found at `~/.cache/huggingface/token` (37 bytes) — scp'd to the VM's own
  `~/.cache/huggingface/token` so the FULL gated-checkpoint cold start + real smoke tests could run
  (not `--skip-smoke`)

## Cold-start execution (scripts/racketsport/gpu_cold_start.sh)
Ran the script almost exactly per RESET_HANDOFF §7 / the script's own recipe, with two real bugs hit
and worked around at the VM level (repo file NOT edited — out of this lane's file-ownership scope;
flagged below for the manager to fix upstream).

**Attempt 1 (revealed Deviation 2 below):** steps 01-04 all logged "ok", but the venv silently had no
pip (`No module named pip` on every subsequent line). Step 05 (checkpoint fetch) correctly hard-failed
for real (`ModuleNotFoundError: No module named 'huggingface_hub'` → sha256 mismatch → `FATAL`) in 11s
total. This IS the fresh-VM state — not something I introduced.

**Fix applied (VM-level only):** `sudo apt-get install -y python3.10-venv` (confirmed missing via
`dpkg -l`; `python3.10 -m ensurepip --version` failed before, `pip 22.0.2` after) + `rm -rf
body_runtime/body_venv` to force a real rebuild.

**Attempt 2 (real, clean):** steps 01-06, `total_wall_seconds: 257` per the script's own
`cold_start_summary.json` — remarkably close to the documented "proven 258s" baseline (within 1s),
strong evidence this VM faithfully reproduces the documented recipe once the venv gap is patched.
- `01_os_deps`: ok, 0s
- `02_clone_pickleball_repo`: ok, 2s — HEAD `5b9f132ee6baa482a3c3f2a10861eb367c427643`
- `03_clone_fast_sam_3d_body_repo`: ok, 0s — pinned commit `808b53c7d9c26a7e511d31144f1e5efb058e15c9`
- `04_build_body_venv`: ok, 201s — real install this time (`body_venv ready` printed for real,
  3.2GB of stale pip cache purged)
- `05_fetch_and_verify_checkpoint`: ok, 48s — `model.ckpt` + `assets/mhr_model.pt` downloaded via the
  scp'd HF token fallback path, sha256-verified against `models/MANIFEST.json`
- `06_pytest_gpu_regression_smoke`: **script says FAILED, but the underlying evidence is a clean
  PASS** — see Deviation 3 below. 27 items collected, **27 passed, 0 skipped, 0 failed**, in 4.13s,
  including all 3 tests the script's own comment flags as `pytest.importorskip("torch")` skip-risks
  (`test_direct_bucket_model_calls_and_numpy_conversion_run_under_inference_mode`,
  `test_warmup_and_real_synthetic_batches_have_matching_guard_signatures`,
  `test_static_clip_intrinsics_warmup_runs_each_bucket_shape_configured_passes`) — all showed
  `PASSED`, proving torch is real/importable/CUDA-capable in `body_venv`, not silently skipped.
  The script's gate does a literal `grep -q "13 passed"` — stale, because the test file has grown
  from 13 to 27 tests since that check was written; it does not indicate any real breakage.
  `run_step` treats this as a hard failure and `exit 1`s before step 07 ever runs.
- `07_gpu_inference_smoke`: not reached by the script (halted after 06's false failure) — ran manually
  instead (see below), reproducing the script's `step_inference_smoke` body verbatim.

## SMOKE TEST — PASS (manual step-07 replica, full command in
`runs/lanes/gpu_coldstart_20260706/` logs, mirrors `gpu_cold_start.sh` exactly)
- Real SAM-3D-Body forward pass on a synthetic 640x480 frame via
  `scripts/racketsport/run_sam3dbody_batch.py` through `scripts/gpu-eval-run.sh`'s lease wrapper:
  `dinov3_vith16plus` backbone (0.257s) → MHR decoder (2.542s) → `[run_inference] TOTAL: 2.916s`;
  script-level timing JSON: `model_setup_load_s=24.96, steady_inference_s=2.92, total_s=29.14`.
- Exit code 0.
- `out.json` structurally verified: `request_count==1`, correct `request_id`, and the single record
  contains `pred_keypoints_3d` (70 keypoints) plus `pred_vertices`, `mesh_faces`, `pred_cam_t`,
  `body_pose_params`, etc. — a fully populated BODY-stage output, not a stub.
- **GPU utilization evidence** (0.5s-interval `nvidia-smi` sampling across the whole call):
  memory climbed 0 MiB → 430 MiB (model onto GPU) → spiked to 1106/1758/2690/3986/4126/4132 MiB with
  utilization ticks of 28%, 13%, 13%, 7%, 3%, 9% landing exactly inside the compute window
  (19:15:54–19:16:02) — real compute activity, not a CPU fallback or mocked response.
- Post-run: GPU back to idle (0%, 0 MiB), `Compute Mode: Exclusive_Process`, no stray python/torch
  processes left running.

## Vendor pins restored (third_party/VENDOR_PINS.md, in the VM's cloned repo)
Ran concurrently with the main cold-start (repo already cloned by step 02) to use wall time
efficiently. All 4 SHAs verified exact matches post-clone:
- `third_party/TOTNet` @ `8a757f63391b262c14d18b4095486336852dbeef`
- `third_party/TrackNetV4` @ `cb7eea7988474771ceac7e880bbffc35bfa87bca`
- `third_party/WASB-SBDT` @ `923462cacdeb3353b84ddebdedb3f4b7a8553b0f`
- `third_party/blurball` @ `2f0f5496f7ba4b5b1a36790749935121b2ce972d`
Then overlaid pickleball's local additions per `third_party/pickleball_vendor_additions/RESTORE.md`
(`cp -R .../WASB-SBDT/* third_party/WASB-SBDT/` and same for blurball) — confirmed
`src/datasets/pickleball.py` present in both after overlay. Note: `third_party/VENDOR_PINS.md` only
documents these 4; three other gitlinks in the repo (`Fast-SAM-3D-Body`, `SAT-HMR`, `TrackNetV3`) exist
too but are outside that doc's table and outside gpu_cold_start.sh's own needs (it clones its own
separate Fast-SAM-3D-Body copy under `body_runtime/`) — left unrestored on this VM as out of scope for
this lane; not needed for the BODY smoke that was the mission's gate.

## Deviations from the §7 / gpu_cold_start.sh recipe (for the manager)
1. **Image family substitution (forced).** `pytorch-latest-gpu` (named in the mission) no longer
   exists in `deeplearning-platform-release` — confirmed via `describe-from-family` (404). Substituted
   the current equivalent `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` (Ubuntu 22.04, required — 24.04
   only ships python3.12 and `step_os_deps` needs apt `python3.10`; driver 580 baked into the image
   name, no runtime driver install needed). Recommend updating any doc that still says
   `pytorch-latest-gpu` to name a currently-real family (or resolve it dynamically at create time).
2. **Real bug: `step_os_deps`'s venv-package detection is a false negative** (found on this exact
   DLVM family, likely on any DLVM image since they ship their own conda stack rather than a
   fully-apt-provisioned Python). It tests `python3.10 -c "import venv"`, which succeeds even when the
   OS-level `python3.10-venv` package (which provides `ensurepip`) is absent — the stdlib `venv` module
   is always importable regardless. Consequence: `python3.10 -m venv` silently creates a **pip-less
   venv**, and **`step_build_body_venv` never notices** — see next.
3. **Real bug: `step_build_body_venv` swallows every internal command failure and always reports
   "ok."** It's invoked as `if "$@"` inside `run_step`, and bash's `set -e` is well-known to be
   suspended for the entire dynamic extent of a function called as an `if`/`while` condition —
   including every command inside that function, transitively. The function's last line is `pip cache
   purge || true`, so it always returns 0 regardless of whether `pip install torch...`,
   `pip install -r requirements`, `pip install chumpy`, or even its own `importlib.util.find_spec`
   sanity heredoc actually succeeded. On this run, ALL of those failed with `No module named pip`
   and the step still logged `ok` in 1s. The failure only surfaced one step later, at
   `05_fetch_and_verify_checkpoint`, as a confusing sha256-mismatch `FATAL` that has nothing
   semantically to do with the real cause. **Recommend:** add an explicit post-creation check inside
   `step_build_body_venv` (e.g. `"$BODY_VENV_DIR/bin/python" -m pip --version || return 1` right after
   `python3.10 -m venv`) so a broken venv fails loud at the right step with the right message.
   Fixed at the VM level only (`sudo apt-get install -y python3.10-venv` + `rm -rf body_venv` + rerun)
   — did not edit the script itself (out of this lane's file-ownership scope).
4. **Real bug: `step_pytest_smoke`'s pass-count gate is stale.**
   `tests/racketsport/test_run_sam3dbody_batch.py` now has 27 tests, not the 13 the script's comment
   and `grep -q "13 passed"` check assume. All 27 passed, 0 skipped, 0 failed — genuinely green — but
   the literal-string gate makes the script report a false `FAILED` and halt before step 07.
   **Recommend:** change the check to something count-agnostic, e.g.
   `grep -qE '^[0-9]+ passed( in| $)' ... && ! grep -qE '[0-9]+ (failed|error)' ...` (and keep the
   existing 0-skipped check as-is). Also update the step's docstring/comment off the hardcoded "13".
   Not fixed in the script itself (out of scope); worked around by manually running step 07's exact
   command body (verbatim reproduction, not a different test) — see SMOKE TEST above.
5. **Auth deviation (intentional, per the top-level mission).** `.claude/skills/gpu-fleet-provision/
   SKILL.md` calls for a service-account key at `~/.secrets/pickleball-fleet-sa.json` and treats a
   missing key as a hard STOP. That file does not exist on this machine. The mission explicitly
   pre-verified and directed use of the already-working interactive `hello@swayformations.com` auth
   instead, so that's what this lane used throughout (create + all SSH/scp). Flagging so the manager
   can decide whether to provision that service-account key for future lanes per the skill's own
   preferred path, or formally bless interactive auth for this project.
6. **`scripts/fleet/lane_vm_startup.sh` and `scripts/fleet/reconcile.sh` are still scaffolds**
   (their own header says "STATUS: SCAFFOLD — flesh out in the P0-1 GPU cold-start lane"). This lane
   used `lane_vm_startup.sh` as-is (it already does the one thing that matters —
   `EXCLUSIVE_PROCESS` + a preemption watcher — and confirmed both work) but did NOT flesh out step 3
   in its own comment (git clone + vendor pins + weights on every boot) or implement `reconcile.sh`,
   because doing so means editing files outside this lane's ownership
   (`runs/lanes/gpu_coldstart_20260706/`, `runs/manager/gpu_fleet.md`, and the VM only). Flagging as a
   clear NEXT action, not silently skipping it.

## NEXT actions for the manager
- Decide on deviations 2-4 (real script bugs) — recommend a follow-up patch to
  `scripts/racketsport/gpu_cold_start.sh` (loud pip-check in step 04, count-agnostic gate in step 06)
  since the NEXT fresh cold start on any DLVM-family image will hit the exact same false-"ok"/false-
  "FAILED" pair.
- `scripts/fleet/lane_vm_startup.sh` / `reconcile.sh` scaffolds still need fleshing out (explicitly
  named as this lane's job in their own header comments, but doing so is outside this lane's declared
  file-ownership boundary — a scope call for the manager, not a decision made unilaterally here).
- VM `pickleball-a100-fleet1` is RUNNING + idle, fully cold-started (repo + vendor pins + body_venv +
  checkpoint all in place at `~/coldstart_20260706/` on the VM), ready for the manager to dispatch the
  next BODY-stage (or general GPU) lane onto it immediately — no re-provisioning needed.
- Only 4 of 7 third_party gitlinks were restored (the ones VENDOR_PINS.md documents); if a future lane
  needs `SAT-HMR`/`TrackNetV3`/the third_party copy of `Fast-SAM-3D-Body` on this VM, restore those too
  (their pinned SHAs are visible via `git ls-tree HEAD third_party/` in the main repo).
- Owner action still pending per RESET_HANDOFF §8: `gcloud compute instances delete
  pickleball-a100-spot-ase1a --project gifted-electron-498923-h1 --zone asia-southeast1-a` (old,
  powered-off VM; unrelated to this lane's new VM, not touched here).

## Ownership discipline
This lane touched only: the new VM (`pickleball-a100-fleet1`, its boot disk, and its in-VM working
directories `~/coldstart_20260706/`, `~/gpu_cold_start.sh`, `~/inference_smoke_manual.sh`,
`~/.cache/huggingface/token`), `runs/manager/gpu_fleet.md`, and
`runs/lanes/gpu_coldstart_20260706/`. No other repo file was modified, including
`scripts/racketsport/gpu_cold_start.sh` and `scripts/fleet/*.sh` despite finding real bugs in them.
