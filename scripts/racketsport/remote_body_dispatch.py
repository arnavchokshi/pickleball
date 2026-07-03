#!/usr/bin/env python3
"""Remote A100 BODY-stage dispatch helper.

`process_video.py`'s SAM3D-only BODY stage needs a CUDA GPU. Most machines running
`process_video.py` (a laptop, a CI box) do not have one, so this module
automates what was previously a manual runbook step: syncing the minimal
inputs a BODY run needs up to the shared A100, running the existing
`threed.racketsport.orchestrator` `--stage body` CLI there under the
project's GPU-lock discipline (`scripts/gpu-eval-run.sh`, a *shared* slot
lease -- never `gpu-train-lock.sh`, which is reserved for exclusive training
jobs and would make BODY dispatch block behind any concurrent training run
indefinitely), and syncing the produced artifacts back.

This module never invents a BODY result: if SSH is unreachable, the shared
GPU lock cannot be acquired within ``lock_wait_timeout_s``, or the remote
command fails, it raises :class:`RemoteBodyDispatchError` with the real
reason. Callers (``process_video.py``) are expected to catch that and
degrade to the already-available skeleton-only bundle rather than crash the
whole pipeline. SAM3D body-mode frames are scheduled before this module is
ever invoked; dispatch only moves those inputs to the A100 and returns the
artifacts produced by the BODY stage.

Nothing here re-runs a gate or changes model internals; it only moves files
and invokes the existing, already-tested orchestrator CLI on a remote host.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# All default remote paths are rooted under one canonical remote home
# directory (review_diff_20260702.md finding 7: "remote BODY dispatch
# default paths are VM-layout fragile"). Before this, DEFAULT_REMOTE_REPO,
# DEFAULT_REMOTE_PYTHON, and the Fast-SAM defaults each hardcoded their own
# copy of "/home/arnavchokshi", so a VM rebuild or user change required
# hunting down and updating every constant independently, and could easily
# leave one stale. They still intentionally point at different subtrees (a
# shared venv for the orchestrator CLI vs. Fast-SAM-3D-Body's own separate
# venv/checkout under body_runtime/) -- that split is real, not a bug -- but
# now they all derive from a single root constant to update.
DEFAULT_REMOTE_HOME = "/home/arnavchokshi"
DEFAULT_REMOTE_HOST = "arnavchokshi@34.126.67.233"
DEFAULT_SSH_KEY = "~/.ssh/google_compute_engine"
DEFAULT_REMOTE_REPO = f"{DEFAULT_REMOTE_HOME}/pickleball_train_main"
# This venv (not FAST_SAM's own) has ultralytics+torch+cuda+opencv+pydantic
# and matches this repo's dependency set closely enough to run the
# orchestrator CLI end to end; Fast-SAM-3D-Body itself is invoked as a
# subprocess by BodyStageRunner via FAST_SAM_PYTHON, set separately below.
DEFAULT_REMOTE_PYTHON = f"{DEFAULT_REMOTE_HOME}/pickleball_git/.venv/bin/python"
DEFAULT_REMOTE_FAST_SAM_PYTHON = f"{DEFAULT_REMOTE_HOME}/body_runtime/fast_sam_venv/bin/python"
DEFAULT_REMOTE_FAST_SAM_ROOT = f"{DEFAULT_REMOTE_HOME}/body_runtime/Fast-SAM-3D-Body"
DEFAULT_GPU_LOCK_SCRIPT = "scripts/gpu-eval-run.sh"
DEFAULT_SSH_CONNECT_TIMEOUT_S = 12
DEFAULT_LOCK_WAIT_TIMEOUT_S = 60
# Overall wall-clock budget for the remote BODY command itself (model loads +
# SAM3D body-mode batches at scheduled frames),
# separate from the lock-wait timeout above. Task #46: these used to be one
# and the same -- `timeout {lock_wait_timeout_s}s` wrapped the *entire* remote
# orchestrator run, so any real BODY run longer than 60s was SIGKILLed at the
# lock-wait budget and misreported as "shared GPU lock busy" (never observed
# before Task #46 only because the remote always failed fast on the missing
# body_frames/ directory that the frames stage now materializes). The lock
# wait is now enforced inside scripts/gpu-eval-run.sh via its own
# GPU_LOCK_TIMEOUT_S env var (exit 75 on lock-wait timeout); this generous
# outer `timeout` budget only guards against a genuinely hung remote run
# (exit 124).
DEFAULT_COMMAND_TIMEOUT_S = 3600
DEFAULT_RUN_ROOT = "runs/process_video_body_dispatch"
# Pinned host key for DEFAULT_REMOTE_HOST (34.126.67.233), captured via
# ssh-keyscan and cross-checked against this machine's own trusted
# ~/.ssh/known_hosts -- see configs/ssh/a100_known_hosts's header comment
# and review_harden_20260702.md finding 7. Used in place of
# StrictHostKeyChecking=no so a wrong/spoofed host cannot silently receive
# source videos or return fabricated BODY artifacts.
DEFAULT_KNOWN_HOSTS_FILE = str(ROOT / "configs" / "ssh" / "a100_known_hosts")

# Remote clip ids are interpolated into a single shell command string sent
# over SSH (_remote_body_command) and into a remote directory path (mkdir).
# Every token is also shlex-quoted before interpolation (defense in depth),
# but this validation is the primary control: it rejects shell metacharacters
# outright rather than relying solely on quoting to neutralize them -- see
# review_harden_20260702.md finding 8.
CLIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
SAM3D_UPSTREAM_ENV_WHITELIST = frozenset(
    {
        "USE_COMPILE_BACKBONE",
        "DECODER_COMPILE",
        "INTERM_COMPILE",
        "INTERM_SLIM",
        "COMPILE_MODE",
        "MHR_NO_CORRECTIVES",
    }
)


def _validate_clip_id(clip: str) -> str:
    if not clip or not CLIP_ID_PATTERN.match(clip):
        raise RemoteBodyDispatchError(
            f"refusing remote BODY dispatch for unsafe clip id {clip!r}: "
            f"must match {CLIP_ID_PATTERN.pattern!r} (letters, digits, '_', '.', '-' only)"
        )
    return clip

# Inputs a BODY run needs (calibration -> tracking -> pose -> body chain),
# synced up when present locally. body_frames/ is synced separately (it is a
# directory of many small JPEGs, not a single artifact file).
BODY_INPUT_ARTIFACTS: tuple[str, ...] = (
    "remote_body_runner.py",
    "capture_sidecar.json",
    "court_calibration.json",
    "court_zones.json",
    "net_plane.json",
    "court_line_evidence.json",
    "tracks.json",
    "skeleton3d.json",
    "ball_track.json",
    "ball_inflections.json",
    "wrist_velocity_peaks.json",
    "contact_windows.json",
    "frame_compute_plan.json",
    "audio_onsets_v2.json",
    "audio_onsets.json",
    "sam3d_body_mask_prompts.json",
)

SAM3D_MASK_INPUT_DIRS: tuple[str, ...] = (
    "sam3d_body_masks",
    "sam3d_mask_prompts",
    "sam2_body_masklets",
)

# Results a BODY run produces, synced back.
BODY_OUTPUT_ARTIFACTS: tuple[str, ...] = (
    "smpl_motion.json",
    "skeleton3d.json",
    "body_compute_execution.json",
    "body_mesh.json",
    "body_mesh_readiness.json",
    "body_joint_quality.json",
    "body_full_clip_gate.json",
    "body_grounding_quality.json",
    "sam3d_tier2_config.json",
    "sam3d_body_input_prep.json",
    "remote_sam3d_tier2_dispatch_config.json",
    "contact_splice.json",
    "frame_compute_plan.json",
    "wrist_velocity_peaks.json",
    "ball_inflections.json",
    "contact_windows.json",
    "pipeline_run.json",
)

RunFn = Callable[[Sequence[str], float | None], "subprocess.CompletedProcess[str]"]


class RemoteBodyDispatchError(RuntimeError):
    """Raised when the remote BODY dispatch cannot complete for a real reason."""


@dataclass(frozen=True)
class RemoteConfig:
    host: str = DEFAULT_REMOTE_HOST
    ssh_key: str = DEFAULT_SSH_KEY
    repo: str = DEFAULT_REMOTE_REPO
    python: str = DEFAULT_REMOTE_PYTHON
    fast_sam_python: str = DEFAULT_REMOTE_FAST_SAM_PYTHON
    fast_sam_root: str = DEFAULT_REMOTE_FAST_SAM_ROOT
    # BODY runner detector/FOV model selection for the remote run. Both default
    # to "" (disabled) because that is the only configuration that has actually
    # produced real meshes on VM1 (every successful a100_body_video_smoke_* run
    # under runs/body_joint_goal_smoke_20260630T001407/ reports
    # detector_name=""/fov_name="" in its body metrics): the committed
    # BodyStageRunner defaults (detector "yolo" + fov "moge2") hard-fail on VM1
    # because the moge_2_vitl_normal checkpoint does not exist anywhere on that
    # host (its manifest entry still points at a stale H100-container
    # /workspace path), and verify_fast_sam_manifest_assets is intentionally
    # strict. Track bboxes are already supplied per-frame from tracks.json, so
    # the detector is redundant for this dispatch path anyway.
    body_detector_name: str = ""
    body_fov_name: str = ""
    sam3d_body_input_size_px: int = 384
    sam3d_crop_bucket_sizes: tuple[int, ...] = (8, 16)
    sam3d_crop_padding_scale: float = 1.0
    sam3d_mask_prompt_mode: str = "manifest"
    sam3d_mask_prompt_artifact: str = "sam3d_body_mask_prompts.json"
    sam3d_soft_background_alpha: float = 1.0
    sam3d_torch_compile: bool = True
    sam3d_compile_warmup_buckets: tuple[int, ...] = (8, 16)
    sam3d_compile_warmup_passes: int = 2
    sam3d_skip_tier2_mesh_vertices: bool = True
    sam3d_steady_state_empty_cache: bool = True
    sam3d_inner_bucket_sync: bool = True
    sam3d_upstream_env: dict[str, str] = field(default_factory=dict)
    sam3d_tier2_output_lite: bool = False
    sam3d_wrist_bone_lock: bool = True
    gpu_lock_script: str = DEFAULT_GPU_LOCK_SCRIPT
    run_root: str = DEFAULT_RUN_ROOT
    connect_timeout_s: int = DEFAULT_SSH_CONNECT_TIMEOUT_S
    lock_wait_timeout_s: int = DEFAULT_LOCK_WAIT_TIMEOUT_S
    command_timeout_s: int = DEFAULT_COMMAND_TIMEOUT_S
    # Pinned known_hosts file for host-key verification (finding 7). Empty
    # string means "no pin configured" and falls back to the caller's
    # regular ~/.ssh/known_hosts with strict checking still on -- it does
    # NOT mean unverified, unlike the old StrictHostKeyChecking=no default.
    known_hosts_file: str = DEFAULT_KNOWN_HOSTS_FILE

    def ssh_option_args(self) -> list[str]:
        """`-o ...` host-key-verification options, shared by ssh and rsync -e."""

        args = [
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout_s}",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "IdentitiesOnly=yes",
        ]
        if self.known_hosts_file:
            args += ["-o", f"UserKnownHostsFile={self.known_hosts_file}"]
        return args

    def ssh_base(self) -> list[str]:
        return ["ssh", "-i", self.ssh_key, *self.ssh_option_args(), self.host]

    def rsync_ssh_command(self) -> str:
        """The `-e` argument rsync uses to shell out to ssh for transport.

        Built from shlex-quoted tokens even though every token here comes
        from RemoteConfig (not directly from user/clip-controlled strings),
        as defense in depth matching finding 8's quoting requirement.
        """

        parts = ["ssh", "-i", self.ssh_key, *self.ssh_option_args()]
        return " ".join(shlex.quote(part) for part in parts)


@dataclass
class RemoteBodyDispatchResult:
    status: str  # "ran" | "lock_busy" | "unreachable" | "failed"
    remote_run_dir: str
    synced_inputs: list[str] = field(default_factory=list)
    synced_outputs: list[str] = field(default_factory=list)
    wall_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    stdout_tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "remote_run_dir": self.remote_run_dir,
            "synced_inputs": list(self.synced_inputs),
            "synced_outputs": list(self.synced_outputs),
            "wall_seconds": round(self.wall_seconds, 3),
            "notes": list(self.notes),
            "stdout_tail": self.stdout_tail,
        }


def _run(cmd: Sequence[str], timeout_s: float | None = None) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def check_remote_reachable(config: RemoteConfig, *, run: RunFn = _run) -> bool:
    """Cheap, side-effect-free SSH reachability probe."""

    try:
        result = run([*config.ssh_base(), "true"], config.connect_timeout_s + 5)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _remote_layout_checks(config: RemoteConfig) -> list[tuple[str, str]]:
    """Remote paths that must exist before mkdir/rsync/dispatch starts.

    review_diff_20260702.md finding 7: the default remote repo, python
    interpreter, gpu lock script, and Fast-SAM-3D-Body paths can silently
    drift out of sync with a given VM's actual layout (a rebuild, a
    different user, a moved venv). Order matters here: the repo root is
    checked first because a missing repo makes every other check (which is
    relative to it, or invoked from inside it) meaningless, and
    `_remote_layout_preflight_command` stops at the first missing path
    (fails fast) so the exact first blocker is what callers see.
    """

    return [
        ("remote repo", config.repo),
        ("remote python interpreter", config.python),
        ("gpu lock script", f"{config.repo}/{config.gpu_lock_script}"),
        ("Fast-SAM-3D-Body python interpreter", config.fast_sam_python),
        ("Fast-SAM-3D-Body root", config.fast_sam_root),
    ]


def _remote_layout_preflight_command(config: RemoteConfig) -> str:
    """One `test -e` chain, in order, stopping at the first missing path.

    Each check is `test -e <path> || { echo MISSING:<label>:<path>; exit 7; }`
    joined with `&&`; a missing path both prints an unambiguous, greppable
    marker line to stdout and exits the whole remote command immediately
    (the `exit 7` runs in the current shell, not a subshell, since `{ }` --
    not `( )` -- is used), so later checks never run once one has failed.
    """

    parts = []
    for label, path in _remote_layout_checks(config):
        message = f"MISSING:{label}:{path}"
        parts.append(f"test -e {shlex.quote(path)} || {{ echo {shlex.quote(message)}; exit 7; }}")
    return " && ".join(parts)


def check_remote_layout(config: RemoteConfig, *, run: RunFn = _run) -> None:
    """Fail fast, before any mkdir/rsync, if the remote VM layout doesn't
    match ``config``.

    Without this, a stale/wrong remote root previously surfaced either as a
    confusing single-file rsync failure ("no such file or directory" for
    whichever artifact happened to sync first) or as a generic non-zero exit
    deep inside the remote orchestrator command -- neither names the actual
    missing path. This runs one SSH round trip that checks every path
    ``_remote_layout_checks`` returns and raises :class:`RemoteBodyDispatchError`
    naming the exact missing path, so the degrade note callers (like
    ``process_video.py``, which just does ``f"...: {exc}"`` into
    ``PIPELINE_SUMMARY.json``) surface is directly actionable.
    """

    remote_cmd = _remote_layout_preflight_command(config)
    result = run([*config.ssh_base(), remote_cmd], config.connect_timeout_s + 10)
    if result.returncode == 0:
        return

    missing_label = missing_path = ""
    for line in (result.stdout or "").splitlines():
        if line.startswith("MISSING:"):
            _, _, rest = line.partition("MISSING:")
            missing_label, _, missing_path = rest.partition(":")
            break

    if missing_path:
        raise RemoteBodyDispatchError(
            f"remote VM layout preflight failed on {config.host}: {missing_label} not found at "
            f"{missing_path!r}. The canonical remote root constants in remote_body_dispatch.py "
            "(DEFAULT_REMOTE_HOME and the paths derived from it) no longer match this VM's actual "
            "layout -- update RemoteConfig (or --remote-repo/--remote-python/--remote-fast-sam-python/"
            "--remote-fast-sam-root) or fix the VM before retrying BODY dispatch."
        )
    raise RemoteBodyDispatchError(
        f"remote VM layout preflight failed on {config.host} (exit {result.returncode}): "
        f"{(result.stderr or '').strip()[-500:] or (result.stdout or '').strip()[-500:]}"
    )


def dispatch_body_stage(
    *,
    clip: str,
    clip_dir: str | Path,
    video_path: str | Path,
    body_frames_dir: str | Path | None = None,
    config: RemoteConfig | None = None,
    max_frames: int | None = None,
    max_players: int = 4,
    run: RunFn = _run,
) -> RemoteBodyDispatchResult:
    """Dispatch the BODY stage to the remote A100 and sync results back.

    Raises :class:`RemoteBodyDispatchError` when the dispatch cannot
    complete (unreachable host, GPU lock busy past
    ``config.lock_wait_timeout_s``, or a failing remote command). Callers
    should catch this and degrade to skeleton-only rather than treat it as
    a pipeline-fatal error.
    """

    config = config or RemoteConfig()
    clip = _validate_clip_id(clip)
    clip_dir = Path(clip_dir)
    video_path = Path(video_path)
    started = time.monotonic()

    if not check_remote_reachable(config, run=run):
        raise RemoteBodyDispatchError(
            f"remote host {config.host} unreachable over SSH within {config.connect_timeout_s}s "
            "(check network/VPN, VM power state, or ssh key path)"
        )

    # finding 7: verify the VM layout before creating any remote directory
    # or starting rsync. `mkdir -p` below would otherwise happily create a
    # bogus `config.repo` from scratch on a VM where it doesn't actually
    # exist, letting rsync "succeed" into a directory that isn't the real
    # repo before the remote command fails with a much less specific error.
    check_remote_layout(config, run=run)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    remote_run_dir = f"{config.repo}/{config.run_root}/{clip}_{stamp}"

    mkdir_result = run(
        [*config.ssh_base(), f"mkdir -p {shlex.quote(remote_run_dir + '/body_frames')}"],
        config.connect_timeout_s + 10,
    )
    if mkdir_result.returncode != 0:
        raise RemoteBodyDispatchError(f"failed to create remote run dir {remote_run_dir}: {mkdir_result.stderr.strip()}")

    # Task #46: write the generated remote runner script into the clip dir (it
    # lands in the run artifacts for debugging) so _rsync_up ships it with the
    # other inputs; the remote command then executes it in place.
    runner_script = _remote_body_runner_script(
        clip=clip,
        remote_run_dir=remote_run_dir,
        config=config,
        max_frames=max_frames,
        max_players=max_players,
    )
    (clip_dir / REMOTE_BODY_RUNNER_FILENAME).write_text(runner_script, encoding="utf-8")

    synced_inputs = _rsync_up(clip_dir, video_path, body_frames_dir, remote_run_dir, config, run=run)

    remote_cmd = _remote_body_command(
        remote_run_dir=remote_run_dir,
        config=config,
    )
    try:
        # Local subprocess guard slightly above the remote `timeout` budget so
        # a dead SSH transport cannot hang dispatch forever; the remote-side
        # `timeout {command_timeout_s}s` (exit 124) is the primary bound.
        command_result = run([*config.ssh_base(), remote_cmd], config.command_timeout_s + 120)
    except subprocess.TimeoutExpired as exc:
        raise RemoteBodyDispatchError(
            f"remote BODY command on {config.host} produced no result within "
            f"{config.command_timeout_s + 120}s (SSH transport hung past the remote-side "
            f"timeout {config.command_timeout_s}s budget)"
        ) from exc
    stdout_tail = "\n".join((command_result.stdout or "").splitlines()[-40:])

    if command_result.returncode == 75:
        # scripts/gpu-eval-run.sh's own GPU_LOCK_TIMEOUT_S flock timeout.
        raise RemoteBodyDispatchError(
            f"shared GPU lock busy on {config.host}: did not acquire {config.gpu_lock_script} within "
            f"{config.lock_wait_timeout_s}s (another job likely holds scripts/gpu-train-lock.sh's exclusive lock)"
        )
    if command_result.returncode == 124:
        raise RemoteBodyDispatchError(
            f"remote BODY run on {config.host} exceeded its overall {config.command_timeout_s}s "
            f"command budget and was killed (raise --remote-command-timeout-s if the run is "
            f"legitimately long); this is NOT the shared-GPU-lock wait, which is bounded separately "
            f"at {config.lock_wait_timeout_s}s via GPU_LOCK_TIMEOUT_S"
        )
    if command_result.returncode != 0:
        raise RemoteBodyDispatchError(
            f"remote BODY stage failed (exit {command_result.returncode}): "
            f"{(command_result.stderr or '').strip()[-2000:] or stdout_tail}"
        )

    synced_outputs = _rsync_down(remote_run_dir, clip_dir, config, run=run)
    if "smpl_motion.json" not in synced_outputs and "skeleton3d.json" not in synced_outputs:
        raise RemoteBodyDispatchError(
            f"remote BODY command exited 0 but produced no smpl_motion.json/skeleton3d.json in {remote_run_dir}"
        )

    return RemoteBodyDispatchResult(
        status="ran",
        remote_run_dir=remote_run_dir,
        synced_inputs=synced_inputs,
        synced_outputs=synced_outputs,
        wall_seconds=time.monotonic() - started,
        notes=[f"dispatched BODY stage to {config.host}:{remote_run_dir} under {config.gpu_lock_script} (shared slot lease)"],
        stdout_tail=stdout_tail,
    )


REMOTE_BODY_RUNNER_FILENAME = "remote_body_runner.py"


def build_phase_d_sam3d_dispatch_config(config: RemoteConfig) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "racketsport_remote_sam3d_tier2_dispatch_config",
        "source": "sam3d_tier2_impl_20260703T0xZ",
        "phase_d_source": "phase_d_speed_opt_20260703T0xZ",
        "body_stage": {
            "skeleton_source": "sam3d_body_joints",
            "tier2_body_joints_all_tracked": True,
            "tier1_mesh_policy": "ball_aware_100",
            "legacy_pose_path": "removed_from_remote_sam3d_dispatch",
        },
        "optimization": {
            "sam3d_body_input_size_px": int(config.sam3d_body_input_size_px),
            "crop_bucket_sizes": [int(value) for value in config.sam3d_crop_bucket_sizes],
            "crop_padding_scale": float(config.sam3d_crop_padding_scale),
            "mask_prompt_mode": str(config.sam3d_mask_prompt_mode),
            "mask_prompt_artifact": str(config.sam3d_mask_prompt_artifact),
            "soft_background_alpha": float(config.sam3d_soft_background_alpha),
            "torch_compile": bool(config.sam3d_torch_compile),
            "compile_warmup_buckets": [int(value) for value in config.sam3d_compile_warmup_buckets],
            "compile_warmup_passes": int(config.sam3d_compile_warmup_passes),
            "steady_state_empty_cache": bool(config.sam3d_steady_state_empty_cache),
            "inner_bucket_sync": bool(config.sam3d_inner_bucket_sync),
            "upstream_env": dict(config.sam3d_upstream_env),
            "tier2_output_lite": bool(config.sam3d_tier2_output_lite),
            "sam3d_wrist_bone_lock": bool(config.sam3d_wrist_bone_lock),
            "batching": "static_intrinsics_cross_frame_bucketed_body_batch",
            "real_batched_execution": {
                "runner": "scripts/racketsport/run_sam3dbody_batch.py",
                "bucket_sizes_to_measure": [int(value) for value in config.sam3d_crop_bucket_sizes],
                "tail_padding": "duplicate last real crop to fill the bucket and discard padded outputs",
                "identity_contract": "request_id maps each model output back to the original frame/player crop",
            },
            "static_clip_intrinsics_contract": {
                "source_artifact": "court_calibration.json",
                "request_field": "clip_intrinsics",
                "batch_runner_kwarg": "clip_intrinsics",
                "shape": [1, 3, 3],
                "warmup_bucket_shapes_match_real_execution": True,
                "warmup_passes_per_shape": int(config.sam3d_compile_warmup_passes),
                "per_request_camera_intrinsics_policy": "must_match_or_error",
            },
        },
        "accuracy_opt": {
            "source": "sam3d_accuracy_opt_20260703T0xZ",
            "mask_prompt_fallback": "box_only_when_mask_absent",
            "camera_intrinsics_policy": "static_per_clip_from_court_calibration",
            "crop_resolution_sweep_sizes_px": [384, 448, 512],
        },
        "serialization": {
            "mesh_vertex_serialization_policy": "tier1_only" if config.sam3d_skip_tier2_mesh_vertices else "all",
            "tier2_mesh_vertices_serialized": not bool(config.sam3d_skip_tier2_mesh_vertices),
        },
        "a100_stall_regression_check": {
            "guard_hypothesis": (
                "warmup previously used a hand-built forward_step batch while real buckets used "
                "run_inference with the transformed request batch; warmup and inference must share "
                "the same builder, static per-clip cam_int, and torch.inference_mode context"
            ),
            "max_first_measured_call_after_warmup_s": 1.0,
            "fails_if_first_call_exceeds_s": 2.0,
            "procedure": [
                "set USE_COMPILE=1 before estimator setup",
                "set COMPILE_WARMUP_BATCH_SIZES from compile_warmup_buckets",
                "load court_calibration.json once and build clip_intrinsics.matrix",
                "run compile warmup for each bucket through the same synthetic-bucket builder and run_inference entrypoint as real buckets for compile_warmup_passes passes",
                "record warmup batch_guard_signatures for keys, shapes, dtypes, devices, strides, and grad/inference mode",
                "time the first real post-warmup bucketed body-batch call and fail if it exceeds threshold",
                "record per-bucket ms/person for bucket sizes 8 and 16",
            ],
        },
        "validation": {
            "protected_eval_labels_used": False,
            "gpu_required_for_timing": True,
            "target_ms_per_person_batched": 55.0,
            "internal_val_only": True,
        },
    }


def _remote_body_success_flags(summary: dict[str, Any], *, skeleton_exists: bool) -> dict[str, Any]:
    stages = summary.get("stages", [])
    if not isinstance(stages, list):
        stages = []
    body_stages = [
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("stage") == "body"
    ]
    body_status = str(body_stages[-1].get("status", "")) if body_stages else ""
    body_ran = bool(body_stages) and body_status == "ran"
    body_notes = " ".join(
        str(note)
        for stage in body_stages
        for note in stage.get("notes", ())
    )
    no_sam3d_body_mode_frames = "adaptive BODY schedule contains no SAM3D body-mode frames" in body_notes
    skeleton_level_only = (
        not body_ran
        and bool(skeleton_exists)
        and body_status in {"failed", "blocked", "skipped"}
        and no_sam3d_body_mode_frames
    )
    return {
        "body_ran": body_ran,
        "body_status": body_status,
        "body_notes": body_notes,
        "skeleton_level_only": skeleton_level_only,
        "no_sam3d_body_mode_frames": no_sam3d_body_mode_frames,
        "requires_pose_stage": False,
    }


def _remote_body_runner_script(
    *,
    clip: str,
    remote_run_dir: str,
    config: RemoteConfig,
    max_frames: int | None,
    max_players: int,
) -> str:
    """Generate the python script the remote host executes for the BODY run.

    Task #46: this replaces the previous direct `-m threed.racketsport.orchestrator
    --stage body` CLI invocation, for one reason the CLI cannot express: the
    committed orchestrator's ``_default_runners`` hardcodes
    ``BodyStageRunner()`` with detector "yolo" + fov "moge2", and VM1 has no
    moge_2_vitl_normal checkpoint anywhere (see ``RemoteConfig.body_fov_name``'s
    docstring) -- the only VM1-proven BODY configuration is
    ``detector_name=""``/``fov_name=""``, which requires registering a custom
    runner via ``run_pipeline(runners=...)``. The script is written locally into
    the clip dir (so it lands in the run artifacts for debugging) and rsynced up
    with the other inputs; every interpolated value is embedded via ``repr()``
    (safe python string literals), and the clip id is additionally validated by
    ``_validate_clip_id`` before dispatch ever gets here.

    Exit rule: 0 when the BODY stage's own StageRun reports "ran" (mirroring
    process_video.py's ``_spine_stage_succeeded`` rather than run_pipeline's
    aggregate status, which can be "blocked" for unrelated whole-closure
    readiness reasons even when BODY itself succeeded), OR when BODY failed
    only because the SAM3D-only scheduler reported
    "adaptive BODY schedule contains no SAM3D body-mode frames" while a real
    skeleton3d.json already exists. The printed JSON marker distinguishes the
    two success shapes for the local caller.
    """

    manifest_path = f"{config.repo}/models/MANIFEST.json"
    dispatch_config = build_phase_d_sam3d_dispatch_config(config)
    dispatch_config_json = json.dumps(dispatch_config, indent=2, sort_keys=True)
    return f"""#!/usr/bin/env python
# Generated by scripts/racketsport/remote_body_dispatch.py (Task #46).
import json
import os
import sys

sys.path.insert(0, {config.repo!r})

from threed.racketsport.orchestrator import BodyStageRunner, run_pipeline

remote_dispatch_sam3d_config = json.loads({dispatch_config_json!r})
with open({remote_run_dir + '/remote_sam3d_tier2_dispatch_config.json'!r}, "w", encoding="utf-8") as config_file:
    json.dump(remote_dispatch_sam3d_config, config_file, indent=2, sort_keys=True)
    config_file.write("\\n")

summary = run_pipeline(
    clip={clip!r},
    inputs_dir={remote_run_dir!r},
    run_dir={remote_run_dir!r},
    stage="body",
    sport="pickleball",
    max_frames={max_frames!r},
    tracking_mode="precomputed_tracks",
    tracking_video={remote_run_dir + '/source.mp4'!r},
    manifest_path={manifest_path!r},
    max_players={max_players!r},
    runners={{
        "body": BodyStageRunner(
            manifest_path={manifest_path!r},
            detector_name={config.body_detector_name!r},
            fov_name={config.body_fov_name!r},
            tier2_body_joints_all_tracked=True,
            mesh_vertex_serialization_policy={'tier1_only' if config.sam3d_skip_tier2_mesh_vertices else 'all'!r},
            sam3d_body_input_size_px={int(config.sam3d_body_input_size_px)!r},
            sam3d_crop_bucket_sizes={tuple(int(value) for value in config.sam3d_crop_bucket_sizes)!r},
            sam3d_crop_padding_scale={float(config.sam3d_crop_padding_scale)!r},
            sam3d_mask_prompt_mode={str(config.sam3d_mask_prompt_mode)!r},
            sam3d_mask_prompt_artifact={str(config.sam3d_mask_prompt_artifact)!r},
            sam3d_soft_background_alpha={float(config.sam3d_soft_background_alpha)!r},
            sam3d_torch_compile={bool(config.sam3d_torch_compile)!r},
            sam3d_compile_warmup_buckets={tuple(int(value) for value in config.sam3d_compile_warmup_buckets)!r},
            sam3d_compile_warmup_passes={int(config.sam3d_compile_warmup_passes)!r},
            sam3d_steady_state_empty_cache={bool(config.sam3d_steady_state_empty_cache)!r},
            sam3d_inner_bucket_sync={bool(config.sam3d_inner_bucket_sync)!r},
            sam3d_upstream_env={dict(config.sam3d_upstream_env)!r},
            sam3d_tier2_output_lite={bool(config.sam3d_tier2_output_lite)!r},
            sam3d_wrist_bone_lock={bool(config.sam3d_wrist_bone_lock)!r},
        )
    }},
)
stages = summary["stages"]
body_stages = [stage for stage in stages if stage.get("stage") == "body"]
body_ran = bool(body_stages) and body_stages[-1].get("status") == "ran"
body_notes = " ".join(str(note) for stage in body_stages for note in stage.get("notes", ()))
skeleton_exists = os.path.isfile({remote_run_dir + '/skeleton3d.json'!r})
body_status = str(body_stages[-1].get("status", "")) if body_stages else ""
no_sam3d_body_mode_frames = "adaptive BODY schedule contains no SAM3D body-mode frames" in body_notes
skeleton_level_only = (
    not body_ran
    and skeleton_exists
    and body_status in {{"failed", "blocked", "skipped"}}
    and no_sam3d_body_mode_frames
)
print(json.dumps({{
    "aggregate_status": summary["status"],
    "stages": [[stage.get("stage"), stage.get("status")] for stage in stages],
    "body_ran": body_ran,
    "body_status": body_status,
    "no_sam3d_body_mode_frames": no_sam3d_body_mode_frames,
    "skeleton_level_only": skeleton_level_only,
}}))
raise SystemExit(0 if (body_ran or skeleton_level_only) else 1)
"""


def _remote_body_command(
    *,
    remote_run_dir: str,
    config: RemoteConfig,
) -> str:
    # Every interpolated token is shlex-quoted (finding 8) so a hostile or
    # buggy path value cannot break out of its argument position (the clip id,
    # which feeds remote_run_dir, is additionally validated by
    # _validate_clip_id earlier in dispatch_body_stage).
    q = shlex.quote
    # Task #46 timeout split: GPU_LOCK_TIMEOUT_S bounds only the shared-lock
    # *wait* inside gpu-eval-run.sh (its own flock -w; exit 75 on lock-wait
    # timeout), while the outer `timeout` is the generous overall budget for
    # the real BODY run itself (exit 124 only when the run genuinely hangs).
    # Previously the outer `timeout` used lock_wait_timeout_s (60s) and
    # SIGKILLed any real BODY run mid-inference, misreported as "lock busy".
    return (
        f"cd {q(config.repo)} && "
        f"FAST_SAM_PYTHON={q(config.fast_sam_python)} FAST_SAM_ROOT={q(config.fast_sam_root)} "
        f"GPU_LOCK_TIMEOUT_S={int(config.lock_wait_timeout_s)} "
        f"timeout {int(config.command_timeout_s)}s {q(config.gpu_lock_script)} "
        f"{q(config.python)} {q(remote_run_dir + '/' + REMOTE_BODY_RUNNER_FILENAME)}"
    )


def _rsync_up(
    clip_dir: Path,
    video_path: Path,
    body_frames_dir: str | Path | None,
    remote_run_dir: str,
    config: RemoteConfig,
    *,
    run: RunFn,
) -> list[str]:
    if not (clip_dir / "tracks.json").is_file():
        raise RemoteBodyDispatchError("refusing remote BODY dispatch without a local tracks.json to sync (nothing to run body on)")

    synced: list[str] = []
    rsync_ssh = config.rsync_ssh_command()

    if video_path.is_file():
        result = run(["rsync", "-az", "-e", rsync_ssh, str(video_path), f"{config.host}:{remote_run_dir}/source.mp4"], None)
        if result.returncode != 0:
            raise RemoteBodyDispatchError(f"rsync of source video failed: {result.stderr.strip()}")
        synced.append("source.mp4")

    for name in BODY_INPUT_ARTIFACTS:
        local_path = clip_dir / name
        if not local_path.is_file():
            continue
        result = run(["rsync", "-az", "-e", rsync_ssh, str(local_path), f"{config.host}:{remote_run_dir}/{name}"], None)
        if result.returncode != 0:
            raise RemoteBodyDispatchError(f"rsync of {name} failed: {result.stderr.strip()}")
        synced.append(name)

    frames_dir = Path(body_frames_dir) if body_frames_dir is not None else clip_dir / "body_frames"
    if frames_dir.is_dir() and any(frames_dir.iterdir()):
        result = run(
            ["rsync", "-az", "-e", rsync_ssh, f"{frames_dir}/", f"{config.host}:{remote_run_dir}/body_frames/"],
            None,
        )
        if result.returncode != 0:
            raise RemoteBodyDispatchError(f"rsync of body_frames/ failed: {result.stderr.strip()}")
        synced.append("body_frames/")

    for dirname in SAM3D_MASK_INPUT_DIRS:
        mask_dir = clip_dir / dirname
        if not mask_dir.is_dir() or not any(mask_dir.iterdir()):
            continue
        result = run(
            ["rsync", "-az", "-e", rsync_ssh, f"{mask_dir}/", f"{config.host}:{remote_run_dir}/{dirname}/"],
            None,
        )
        if result.returncode != 0:
            raise RemoteBodyDispatchError(f"rsync of {dirname}/ failed: {result.stderr.strip()}")
        synced.append(f"{dirname}/")

    return synced


def _rsync_down(remote_run_dir: str, clip_dir: Path, config: RemoteConfig, *, run: RunFn) -> list[str]:
    synced: list[str] = []
    rsync_ssh = config.rsync_ssh_command()
    clip_dir.mkdir(parents=True, exist_ok=True)

    for name in BODY_OUTPUT_ARTIFACTS:
        result = run(
            ["rsync", "-az", "-e", rsync_ssh, f"{config.host}:{remote_run_dir}/{name}", str(clip_dir / name)],
            None,
        )
        if result.returncode == 0 and (clip_dir / name).is_file():
            synced.append(name)
    return synced


def _parse_int_tuple(value: str, *, flag_name: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{flag_name} must be a comma-separated list of integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{flag_name} must include at least one integer")
    if any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError(f"{flag_name} values must be positive")
    return parsed


def _parse_sam3d_upstream_env_tuple(value: str) -> dict[str, str]:
    if not value.strip():
        return {}
    parsed: dict[str, str] = {}
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        key, separator, raw_env_value = part.partition("=")
        key = key.strip()
        env_value = raw_env_value.strip()
        if not separator or not key or not env_value:
            raise argparse.ArgumentTypeError("--sam3d-upstream-env entries must be KEY=VALUE pairs")
        if key not in SAM3D_UPSTREAM_ENV_WHITELIST:
            raise argparse.ArgumentTypeError(
                f"unsupported SAM3D upstream env key {key!r}; allowed keys are {sorted(SAM3D_UPSTREAM_ENV_WHITELIST)}"
            )
        parsed[key] = env_value
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dispatch the BODY (Fast SAM-3D-Body) stage to the remote A100.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--clip-dir", type=Path, required=True, help="Local directory with tracks.json/skeleton3d.json/etc.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--body-frames-dir", type=Path, default=None)
    parser.add_argument("--host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    parser.add_argument("--repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--fast-sam-python", default=DEFAULT_REMOTE_FAST_SAM_PYTHON)
    parser.add_argument("--fast-sam-root", default=DEFAULT_REMOTE_FAST_SAM_ROOT)
    parser.add_argument("--lock-wait-timeout-s", type=int, default=DEFAULT_LOCK_WAIT_TIMEOUT_S)
    parser.add_argument(
        "--command-timeout-s",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_S,
        help="Overall wall-clock budget for the remote BODY run itself (separate from --lock-wait-timeout-s).",
    )
    parser.add_argument("--sam3d-body-input-size-px", type=int, default=384, choices=(384, 448, 512))
    parser.add_argument("--sam3d-crop-bucket-sizes", default="8,16")
    parser.add_argument("--sam3d-crop-padding-scale", type=float, default=1.0)
    parser.add_argument("--sam3d-mask-prompt-mode", choices=("off", "manifest"), default="manifest")
    parser.add_argument("--sam3d-soft-background-alpha", type=float, default=1.0)
    parser.add_argument("--sam3d-compile-warmup-buckets", default="8,16")
    parser.add_argument("--sam3d-compile-warmup-passes", type=int, default=2)
    parser.add_argument("--no-sam3d-torch-compile", action="store_true")
    parser.add_argument("--serialize-tier2-mesh-vertices", action="store_true")
    parser.add_argument("--no-sam3d-steady-state-empty-cache", action="store_true")
    parser.add_argument("--no-sam3d-inner-bucket-sync", action="store_true")
    parser.add_argument("--sam3d-upstream-env", default="")
    parser.add_argument("--sam3d-tier2-output-lite", action="store_true")
    parser.add_argument("--no-sam3d-wrist-bone-lock", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-players", type=int, default=4)
    parser.add_argument(
        "--known-hosts-file",
        default=DEFAULT_KNOWN_HOSTS_FILE,
        help=(
            "Pinned known_hosts file for SSH host-key verification (default: "
            f"{DEFAULT_KNOWN_HOSTS_FILE}). Pass an empty string to fall back to the "
            "regular ~/.ssh/known_hosts (StrictHostKeyChecking stays on either way)."
        ),
    )
    args = parser.parse_args(argv)

    config = RemoteConfig(
        host=args.host,
        ssh_key=args.ssh_key,
        repo=args.repo,
        python=args.python,
        fast_sam_python=args.fast_sam_python,
        fast_sam_root=args.fast_sam_root,
        sam3d_body_input_size_px=args.sam3d_body_input_size_px,
        sam3d_crop_bucket_sizes=_parse_int_tuple(args.sam3d_crop_bucket_sizes, flag_name="--sam3d-crop-bucket-sizes"),
        sam3d_crop_padding_scale=args.sam3d_crop_padding_scale,
        sam3d_mask_prompt_mode=args.sam3d_mask_prompt_mode,
        sam3d_soft_background_alpha=args.sam3d_soft_background_alpha,
        sam3d_torch_compile=not args.no_sam3d_torch_compile,
        sam3d_compile_warmup_buckets=_parse_int_tuple(
            args.sam3d_compile_warmup_buckets,
            flag_name="--sam3d-compile-warmup-buckets",
        ),
        sam3d_compile_warmup_passes=args.sam3d_compile_warmup_passes,
        sam3d_skip_tier2_mesh_vertices=not args.serialize_tier2_mesh_vertices,
        sam3d_steady_state_empty_cache=not args.no_sam3d_steady_state_empty_cache,
        sam3d_inner_bucket_sync=not args.no_sam3d_inner_bucket_sync,
        sam3d_upstream_env=_parse_sam3d_upstream_env_tuple(args.sam3d_upstream_env),
        sam3d_tier2_output_lite=bool(args.sam3d_tier2_output_lite),
        sam3d_wrist_bone_lock=not args.no_sam3d_wrist_bone_lock,
        lock_wait_timeout_s=args.lock_wait_timeout_s,
        command_timeout_s=args.command_timeout_s,
        known_hosts_file=args.known_hosts_file,
    )
    try:
        result = dispatch_body_stage(
            clip=args.clip,
            clip_dir=args.clip_dir,
            video_path=args.video,
            body_frames_dir=args.body_frames_dir,
            config=config,
            max_frames=args.max_frames,
            max_players=args.max_players,
        )
    except RemoteBodyDispatchError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
