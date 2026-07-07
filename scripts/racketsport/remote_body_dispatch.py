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
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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
# FLEET NOTE (2026-07-06): the old standing VM (34.126.67.233) was DELETED; defaults now point at
# fleet GPU #1 `pickleball-a100-fleet1` with the cold-start layout under ~/coldstart_20260706
# (see runs/manager/gpu_fleet.md — THE source of truth; wave-2 improvement booked: derive these
# from the fleet ledger instead of constants). Per-lane dispatches may still override via flags.
DEFAULT_REMOTE_HOME = "/home/arnavchokshi/coldstart_20260706"
DEFAULT_REMOTE_HOST = "arnavchokshi@34.143.175.207"
DEFAULT_SSH_KEY = "~/.ssh/google_compute_engine"
DEFAULT_REMOTE_REPO = f"{DEFAULT_REMOTE_HOME}/repo"
# On fleet1 the cold-start builds ONE body venv that serves both the orchestrator CLI and
# Fast-SAM-3D-Body subprocess (P0-1 lane verified 27/27 GPU tests through it).
DEFAULT_REMOTE_PYTHON = f"{DEFAULT_REMOTE_HOME}/body_runtime/body_venv/bin/python"
DEFAULT_REMOTE_FAST_SAM_PYTHON = f"{DEFAULT_REMOTE_HOME}/body_runtime/body_venv/bin/python"
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
# Pinned host key for DEFAULT_REMOTE_HOST (fleet1, 34.143.175.207), captured via
# ssh-keyscan and cross-checked against this machine's own trusted
# ~/.ssh/known_hosts -- see configs/ssh/a100_known_hosts's header comment
# and review_harden_20260702.md finding 7. Used in place of
# StrictHostKeyChecking=no so a wrong/spoofed host cannot silently receive
# source videos or return fabricated BODY artifacts.
DEFAULT_KNOWN_HOSTS_FILE = str(ROOT / "configs" / "ssh" / "a100_known_hosts")
TRANSPORT_TAR_BATCH = "tar_batch"
TRANSPORT_RSYNC = "rsync"
TRANSPORT_CHOICES = (TRANSPORT_TAR_BATCH, TRANSPORT_RSYNC)
RETRYABLE_TRANSPORT_EXIT_CODES = (10, 12, 30, 35, 255)
DEFAULT_TRANSPORT_RETRY_MAX_ATTEMPTS = 12
DEFAULT_TRANSPORT_RETRY_BACKOFF_S = 4.0

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
    "camera_motion.json",
    "placement.json",
    "foot_contact_phases.json",
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

# Results a BODY run produces. Heavy monoliths are opt-in downloads because the
# VM can spend minutes serializing and transferring multi-GB pretty JSON that
# the local replay path normally does not consume.
BODY_OUTPUT_ARTIFACTS_HEAVY: tuple[str, ...] = (
    "smpl_motion.json",
    "body_mesh.json",
)
BODY_OUTPUT_ARTIFACTS_DEFAULT: tuple[str, ...] = (
    "skeleton3d.json",
    "body_compute_execution.json",
    "body_mesh_readiness.json",
    "body_joint_quality.json",
    "body_full_clip_gate.json",
    "body_grounding_quality.json",
    "body_serialization_timing.json",
    "body_stage_phase_timing.json",
    "sam3d_keypoints_2d.json",
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
BODY_OUTPUT_DIRS_DEFAULT: tuple[str, ...] = (
    "body_mesh_index/",
)
BODY_OUTPUT_ARTIFACTS: tuple[str, ...] = BODY_OUTPUT_ARTIFACTS_DEFAULT + BODY_OUTPUT_ARTIFACTS_HEAVY

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
    fetch_body_monoliths: bool = False
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
    # Default to the tar-batch transport so process_video.py call sites pick up
    # the wave-2 hardening without changing their invocation contract. GNU
    # rsync remains selectable as a fallback through RemoteConfig/CLI.
    transport: str = TRANSPORT_TAR_BATCH
    transport_retry_max_attempts: int = DEFAULT_TRANSPORT_RETRY_MAX_ATTEMPTS
    transport_retry_backoff_s: float = DEFAULT_TRANSPORT_RETRY_BACKOFF_S
    transport_retryable_exit_codes: tuple[int, ...] = RETRYABLE_TRANSPORT_EXIT_CODES

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

    def scp_base(self) -> list[str]:
        return ["scp", "-i", self.ssh_key, *self.ssh_option_args()]

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
    timing: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "remote_run_dir": self.remote_run_dir,
            "synced_inputs": list(self.synced_inputs),
            "synced_outputs": list(self.synced_outputs),
            "wall_seconds": round(self.wall_seconds, 3),
            "notes": list(self.notes),
            "stdout_tail": self.stdout_tail,
            "timing": dict(self.timing),
        }


def _run(cmd: Sequence[str], timeout_s: float | None = None) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


REMOTE_OUTPUT_LOG_MAX_BYTES = 500 * 1024


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def _upload_size_bytes(
    *,
    synced_inputs: Sequence[str],
    clip_dir: Path,
    video_path: Path,
    body_frames_dir: str | Path | None,
    camera_motion_path: str | Path | None,
) -> int:
    frames_dir = Path(body_frames_dir) if body_frames_dir is not None else clip_dir / "body_frames"
    explicit_camera_motion = Path(camera_motion_path) if camera_motion_path is not None else None
    total = 0
    for name in synced_inputs:
        if name == "source.mp4":
            total += _path_size_bytes(video_path)
        elif name == "body_frames/":
            total += _path_size_bytes(frames_dir)
        elif name in {f"{dirname}/" for dirname in SAM3D_MASK_INPUT_DIRS}:
            total += _path_size_bytes(clip_dir / name.rstrip("/"))
        elif (
            name == CAMERA_MOTION_ARTIFACT
            and explicit_camera_motion is not None
            and explicit_camera_motion.is_file()
            and explicit_camera_motion.resolve() != (clip_dir / CAMERA_MOTION_ARTIFACT).resolve()
        ):
            total += _path_size_bytes(explicit_camera_motion)
        else:
            total += _path_size_bytes(clip_dir / name)
    return total


def _download_size_bytes(*, synced_outputs: Sequence[str], clip_dir: Path) -> int:
    return sum(_path_size_bytes(clip_dir / name.rstrip("/")) for name in synced_outputs)


def _tail_bytes_text(text: str, *, max_bytes: int = REMOTE_OUTPUT_LOG_MAX_BYTES) -> str:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return text
    return raw[-max_bytes:].decode("utf-8", errors="replace")


def _write_remote_output_log(clip_dir: Path, *, stdout: str, stderr: str) -> None:
    combined = "\n".join(
        [
            "### remote stdout",
            stdout or "",
            "### remote stderr",
            stderr or "",
        ]
    )
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / "remote_body_stdout.log").write_text(_tail_bytes_text(combined), encoding="utf-8")


@dataclass(frozen=True)
class _BodyInputSelection:
    single_files: dict[str, Path]
    directories: dict[str, Path]

    @property
    def synced_names(self) -> list[str]:
        return [*self.single_files.keys(), *self.directories.keys()]


def _validate_transport_name(transport: str) -> str:
    if transport not in TRANSPORT_CHOICES:
        raise RemoteBodyDispatchError(
            f"unsupported remote BODY transport {transport!r}; expected one of {', '.join(TRANSPORT_CHOICES)}"
        )
    return transport


def _transport_failure_detail(result: "subprocess.CompletedProcess[str]") -> str:
    return (result.stderr or result.stdout or "").strip()[-2000:]


def _run_transport_command(
    cmd: Sequence[str],
    timeout_s: float | None,
    *,
    config: RemoteConfig,
    run: RunFn,
    operation: str,
    tolerated_failure: Callable[["subprocess.CompletedProcess[str]"], bool] | None = None,
) -> "subprocess.CompletedProcess[str]":
    attempts = max(1, int(config.transport_retry_max_attempts))
    retryable = set(config.transport_retryable_exit_codes)
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        last = run(cmd, timeout_s)
        if last.returncode == 0:
            return last
        if tolerated_failure is not None and tolerated_failure(last):
            return last
        if last.returncode not in retryable:
            raise RemoteBodyDispatchError(
                f"{operation} failed (exit {last.returncode}): {_transport_failure_detail(last)}"
            )
        if attempt < attempts:
            time.sleep(float(config.transport_retry_backoff_s))
    assert last is not None
    raise RemoteBodyDispatchError(
        f"{operation} failed after {attempts} attempts (last exit {last.returncode}): "
        f"{_transport_failure_detail(last)}"
    )


def _collect_body_inputs(
    clip_dir: Path,
    video_path: Path,
    body_frames_dir: str | Path | None,
    *,
    camera_motion_path: str | Path | None = None,
) -> _BodyInputSelection:
    if not (clip_dir / "tracks.json").is_file():
        raise RemoteBodyDispatchError("refusing remote BODY dispatch without a local tracks.json to sync (nothing to run body on)")

    single_file_inputs: dict[str, Path] = {}
    if video_path.is_file():
        single_file_inputs["source.mp4"] = video_path

    explicit_camera_motion = Path(camera_motion_path) if camera_motion_path is not None else None
    for name in BODY_INPUT_ARTIFACTS:
        if name == CAMERA_MOTION_ARTIFACT and explicit_camera_motion is not None and explicit_camera_motion.is_file():
            continue
        local_path = clip_dir / name
        if local_path.is_file():
            single_file_inputs[name] = local_path

    if explicit_camera_motion is not None and explicit_camera_motion.is_file():
        canonical_clip_camera_motion = clip_dir / CAMERA_MOTION_ARTIFACT
        if not canonical_clip_camera_motion.is_file() or explicit_camera_motion.resolve() != canonical_clip_camera_motion.resolve():
            single_file_inputs[CAMERA_MOTION_ARTIFACT] = explicit_camera_motion

    directories: dict[str, Path] = {}
    frames_dir = Path(body_frames_dir) if body_frames_dir is not None else clip_dir / "body_frames"
    if frames_dir.is_dir() and any(frames_dir.iterdir()):
        directories["body_frames/"] = frames_dir

    for dirname in SAM3D_MASK_INPUT_DIRS:
        mask_dir = clip_dir / dirname
        if mask_dir.is_dir() and any(mask_dir.iterdir()):
            directories[f"{dirname}/"] = mask_dir

    return _BodyInputSelection(single_files=single_file_inputs, directories=directories)


def _safe_tar_member_name(name: str) -> str:
    normalized = name.strip().rstrip("/")
    if not normalized or normalized in {".", ".."}:
        raise RemoteBodyDispatchError(f"refusing unsafe tar member name {name!r}")
    parts = Path(normalized).parts
    if normalized.startswith("/") or ".." in parts:
        raise RemoteBodyDispatchError(f"refusing unsafe tar member name {name!r}")
    return normalized


def _add_path_to_tar(archive: tarfile.TarFile, source_path: Path, arcname: str) -> None:
    safe_arcname = _safe_tar_member_name(arcname)
    if source_path.is_dir():
        for path in sorted(source_path.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=f"{safe_arcname}/{path.relative_to(source_path)}", recursive=False)
    elif source_path.is_file():
        archive.add(source_path, arcname=safe_arcname, recursive=False)


def _safe_extract_tar_gz(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_root = dest_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            if member.issym() or member.islnk():
                raise RemoteBodyDispatchError(f"refusing to extract tar link member {member.name!r}")
            target = (dest_dir / member.name).resolve()
            if target != dest_root and not str(target).startswith(str(dest_root) + os.sep):
                raise RemoteBodyDispatchError(f"refusing to extract tar member outside destination: {member.name!r}")
        archive.extractall(dest_dir, members=members)


def _runner_marker_epoch(stdout: str, event: str) -> float | None:
    for line in (stdout or "").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("event") != event:
            continue
        try:
            return float(payload["epoch_s"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _download_artifacts_for_config(config: RemoteConfig) -> tuple[str, ...]:
    if config.fetch_body_monoliths:
        return BODY_OUTPUT_ARTIFACTS_DEFAULT + BODY_OUTPUT_ARTIFACTS_HEAVY
    return BODY_OUTPUT_ARTIFACTS_DEFAULT


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
    camera_motion_path: str | Path | None = None,
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
    transport = _validate_transport_name(config.transport)
    clip = _validate_clip_id(clip)
    clip_dir = Path(clip_dir)
    video_path = Path(video_path)
    started = time.monotonic()
    phase_seconds: dict[str, float] = {}
    upload_bytes = 0
    download_bytes = 0
    lock_wait_estimate_s: float | None = None

    preflight_started = time.monotonic()
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
    phase_seconds["preflight_s"] = time.monotonic() - preflight_started

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    remote_run_dir = f"{config.repo}/{config.run_root}/{clip}_{stamp}"

    mkdir_started = time.monotonic()
    mkdir_result = run(
        [*config.ssh_base(), f"mkdir -p {shlex.quote(remote_run_dir + '/body_frames')}"],
        config.connect_timeout_s + 10,
    )
    phase_seconds["mkdir_s"] = time.monotonic() - mkdir_started
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

    upload_started = time.monotonic()
    transport_phases: dict[str, float] = {}
    synced_inputs = _sync_body_inputs(
        clip_dir,
        video_path,
        body_frames_dir,
        remote_run_dir,
        config,
        run=run,
        camera_motion_path=camera_motion_path,
        phases=transport_phases,
    )
    phase_seconds["upload_s"] = time.monotonic() - upload_started
    phase_seconds.update(transport_phases)
    upload_bytes = _upload_size_bytes(
        synced_inputs=synced_inputs,
        clip_dir=clip_dir,
        video_path=video_path,
        body_frames_dir=body_frames_dir,
        camera_motion_path=camera_motion_path,
    )

    remote_cmd = _remote_body_command(
        remote_run_dir=remote_run_dir,
        config=config,
    )
    remote_command_started = time.monotonic()
    remote_command_started_epoch = time.time()
    try:
        # Local subprocess guard slightly above the remote `timeout` budget so
        # a dead SSH transport cannot hang dispatch forever; the remote-side
        # `timeout {command_timeout_s}s` (exit 124) is the primary bound.
        command_result = run([*config.ssh_base(), remote_cmd], config.command_timeout_s + 120)
    except subprocess.TimeoutExpired as exc:
        phase_seconds["remote_command_s"] = time.monotonic() - remote_command_started
        _write_remote_output_log(
            clip_dir,
            stdout="",
            stderr=f"remote BODY ssh subprocess timed out after {config.command_timeout_s + 120}s",
        )
        timing = {
            "schema_version": 1,
            "artifact_type": "racketsport_remote_body_dispatch_timing",
            "status": "failed",
            "transport": transport,
            "remote_host": config.host,
            "remote_run_dir": remote_run_dir,
            "phases": {name: round(value, 6) for name, value in phase_seconds.items()},
            "upload_bytes": int(upload_bytes),
            "download_bytes": 0,
            "lock_wait_estimate_s": None,
        }
        _write_json_file(clip_dir / "remote_body_dispatch_timing.json", timing)
        raise RemoteBodyDispatchError(
            f"remote BODY command on {config.host} produced no result within "
            f"{config.command_timeout_s + 120}s (SSH transport hung past the remote-side "
            f"timeout {config.command_timeout_s}s budget)"
        ) from exc
    phase_seconds["remote_command_s"] = time.monotonic() - remote_command_started
    _write_remote_output_log(clip_dir, stdout=command_result.stdout or "", stderr=command_result.stderr or "")
    stdout_tail = "\n".join((command_result.stdout or "").splitlines()[-40:])
    runner_start_epoch = _runner_marker_epoch(command_result.stdout or "", "script_start")
    if runner_start_epoch is not None:
        # This is an estimate: it includes SSH command startup and remote shell
        # wrapper overhead before the runner's first Python marker prints.
        lock_wait_estimate_s = runner_start_epoch - remote_command_started_epoch

    if command_result.returncode == 75:
        timing = {
            "schema_version": 1,
            "artifact_type": "racketsport_remote_body_dispatch_timing",
            "status": "lock_busy",
            "transport": transport,
            "remote_host": config.host,
            "remote_run_dir": remote_run_dir,
            "phases": {name: round(value, 6) for name, value in phase_seconds.items()},
            "upload_bytes": int(upload_bytes),
            "download_bytes": 0,
            "lock_wait_estimate_s": lock_wait_estimate_s,
        }
        _write_json_file(clip_dir / "remote_body_dispatch_timing.json", timing)
        # scripts/gpu-eval-run.sh's own GPU_LOCK_TIMEOUT_S flock timeout.
        raise RemoteBodyDispatchError(
            f"shared GPU lock busy on {config.host}: did not acquire {config.gpu_lock_script} within "
            f"{config.lock_wait_timeout_s}s (another job likely holds scripts/gpu-train-lock.sh's exclusive lock)"
        )
    if command_result.returncode == 124:
        timing = {
            "schema_version": 1,
            "artifact_type": "racketsport_remote_body_dispatch_timing",
            "status": "timeout",
            "transport": transport,
            "remote_host": config.host,
            "remote_run_dir": remote_run_dir,
            "phases": {name: round(value, 6) for name, value in phase_seconds.items()},
            "upload_bytes": int(upload_bytes),
            "download_bytes": 0,
            "lock_wait_estimate_s": lock_wait_estimate_s,
        }
        _write_json_file(clip_dir / "remote_body_dispatch_timing.json", timing)
        raise RemoteBodyDispatchError(
            f"remote BODY run on {config.host} exceeded its overall {config.command_timeout_s}s "
            f"command budget and was killed (raise --remote-command-timeout-s if the run is "
            f"legitimately long); this is NOT the shared-GPU-lock wait, which is bounded separately "
            f"at {config.lock_wait_timeout_s}s via GPU_LOCK_TIMEOUT_S"
        )
    if command_result.returncode != 0:
        timing = {
            "schema_version": 1,
            "artifact_type": "racketsport_remote_body_dispatch_timing",
            "status": "failed",
            "transport": transport,
            "remote_host": config.host,
            "remote_run_dir": remote_run_dir,
            "phases": {name: round(value, 6) for name, value in phase_seconds.items()},
            "upload_bytes": int(upload_bytes),
            "download_bytes": 0,
            "lock_wait_estimate_s": lock_wait_estimate_s,
        }
        _write_json_file(clip_dir / "remote_body_dispatch_timing.json", timing)
        raise RemoteBodyDispatchError(
            f"remote BODY stage failed (exit {command_result.returncode}): "
            f"{(command_result.stderr or '').strip()[-2000:] or stdout_tail}"
        )

    download_started = time.monotonic()
    transport_phases = {}
    synced_outputs = _sync_body_outputs(remote_run_dir, clip_dir, config, run=run, phases=transport_phases)
    phase_seconds["download_s"] = time.monotonic() - download_started
    phase_seconds.update(transport_phases)
    download_bytes = _download_size_bytes(synced_outputs=synced_outputs, clip_dir=clip_dir)
    timing = {
        "schema_version": 1,
        "artifact_type": "racketsport_remote_body_dispatch_timing",
        "status": "ran",
        "transport": transport,
        "remote_host": config.host,
        "remote_run_dir": remote_run_dir,
        "phases": {name: round(value, 6) for name, value in phase_seconds.items()},
        "upload_bytes": int(upload_bytes),
        "download_bytes": int(download_bytes),
        "lock_wait_estimate_s": lock_wait_estimate_s,
        "fetched_default_artifacts": list(BODY_OUTPUT_ARTIFACTS_DEFAULT),
        "fetched_heavy_artifacts": list(BODY_OUTPUT_ARTIFACTS_HEAVY) if config.fetch_body_monoliths else [],
        "skipped_heavy_artifacts": [] if config.fetch_body_monoliths else list(BODY_OUTPUT_ARTIFACTS_HEAVY),
        "fetched_directories": [name for name in BODY_OUTPUT_DIRS_DEFAULT if name in synced_outputs],
    }
    _write_json_file(clip_dir / "remote_body_dispatch_timing.json", timing)
    if "smpl_motion.json" not in synced_outputs and "skeleton3d.json" not in synced_outputs:
        raise RemoteBodyDispatchError(
            f"remote BODY command exited 0 but produced no smpl_motion.json/skeleton3d.json in {remote_run_dir}"
        )
    fetch_notes = [
        f"remote BODY phase timing: transport={transport} preflight={phase_seconds.get('preflight_s', 0.0):.3f}s "
        f"mkdir={phase_seconds.get('mkdir_s', 0.0):.3f}s upload={phase_seconds.get('upload_s', 0.0):.3f}s "
        f"remote_command={phase_seconds.get('remote_command_s', 0.0):.3f}s download={phase_seconds.get('download_s', 0.0):.3f}s "
        f"upload_bytes={upload_bytes} download_bytes={download_bytes}",
        (
            "fetched BODY default artifacts plus heavy monoliths because fetch_body_monoliths=True"
            if config.fetch_body_monoliths
            else "fetched BODY default artifacts only; smpl_motion.json and body_mesh.json were not built on the VM (speed default; rerun with --fetch-body-monoliths to produce them)"
        ),
    ]
    if "body_mesh_index/" in synced_outputs:
        fetch_notes.append("fetched body_mesh_index/ windowed mesh directory for replay review; not a BODY verification claim")
    else:
        fetch_notes.append("body_mesh_index/ was not fetched because the remote run did not produce it or rsync reported it absent")

    return RemoteBodyDispatchResult(
        status="ran",
        remote_run_dir=remote_run_dir,
        synced_inputs=synced_inputs,
        synced_outputs=synced_outputs,
        wall_seconds=time.monotonic() - started,
        notes=[
            f"dispatched BODY stage to {config.host}:{remote_run_dir} under {config.gpu_lock_script} (shared slot lease)",
            *fetch_notes,
        ],
        stdout_tail=stdout_tail,
        timing=timing,
    )


REMOTE_BODY_RUNNER_FILENAME = "remote_body_runner.py"
CAMERA_MOTION_ARTIFACT = "camera_motion.json"


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
import time


def _emit_marker(event, **extra):
    payload = {{"event": event, "epoch_s": time.time()}}
    payload.update(extra)
    print(json.dumps(payload, sort_keys=True), flush=True)


_emit_marker("script_start")

import os
import sys

sys.path.insert(0, {config.repo!r})

from threed.racketsport.body_mesh_index import build_body_mesh_index, build_body_mesh_index_cli_summary
from threed.racketsport.orchestrator import BodyStageRunner, run_pipeline

_emit_marker("imports_done")

remote_dispatch_sam3d_config = json.loads({dispatch_config_json!r})
with open({remote_run_dir + '/remote_sam3d_tier2_dispatch_config.json'!r}, "w", encoding="utf-8") as config_file:
    json.dump(remote_dispatch_sam3d_config, config_file, indent=2, sort_keys=True)
    config_file.write("\\n")

os.chdir({remote_run_dir!r})

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
    reuse_existing_stage_artifacts=True,
    runners={{
        "body": BodyStageRunner(
            manifest_path={manifest_path!r},
            detector_name={config.body_detector_name!r},
            fov_name={config.body_fov_name!r},
            tier2_body_joints_all_tracked=True,
            mesh_vertex_serialization_policy={'tier1_only' if config.sam3d_skip_tier2_mesh_vertices else 'all'!r},
            write_body_monoliths={bool(config.fetch_body_monoliths)!r},
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
_emit_marker("run_pipeline_done", aggregate_status=summary["status"])
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
body_mesh_path = {remote_run_dir + '/body_mesh.json'!r}
body_mesh_index_path = {remote_run_dir + '/body_mesh_index/body_mesh_index.json'!r}
if os.path.isfile(body_mesh_index_path):
    _emit_marker("mesh_index_skipped", mesh_index="existing", produced_by="orchestrator_in_memory", reason="body_mesh_index already exists")
elif os.path.isfile(body_mesh_path):
    try:
        mesh_index_result = build_body_mesh_index({remote_run_dir!r}, out_dir={remote_run_dir + '/body_mesh_index'!r})
        _emit_marker("mesh_index_done", mesh_index=build_body_mesh_index_cli_summary(mesh_index_result))
    except Exception as exc:  # noqa: BLE001 - mesh index is an optimization only
        print(json.dumps({{"event": "mesh_index_failed", "epoch_s": time.time(), "mesh_index": "failed", "error": str(exc)}}, sort_keys=True), flush=True)
else:
    _emit_marker("mesh_index_skipped", mesh_index="skipped", reason="body_mesh.json not found", skeleton_level_only=skeleton_level_only)
print(json.dumps({{
    "aggregate_status": summary["status"],
    "stages": [[stage.get("stage"), stage.get("status")] for stage in stages],
    "body_ran": body_ran,
    "body_status": body_status,
    "no_sam3d_body_mode_frames": no_sam3d_body_mode_frames,
    "skeleton_level_only": skeleton_level_only,
}}))
exit_code = 0 if (body_ran or skeleton_level_only) else 1
_emit_marker("exit", exit_code=exit_code)
raise SystemExit(exit_code)
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


def _sync_body_inputs(
    clip_dir: Path,
    video_path: Path,
    body_frames_dir: str | Path | None,
    remote_run_dir: str,
    config: RemoteConfig,
    *,
    run: RunFn,
    phases: dict[str, float] | None = None,
    camera_motion_path: str | Path | None = None,
) -> list[str]:
    transport = _validate_transport_name(config.transport)
    if transport == TRANSPORT_RSYNC:
        return _rsync_up(
            clip_dir,
            video_path,
            body_frames_dir,
            remote_run_dir,
            config,
            run=run,
            camera_motion_path=camera_motion_path,
        )
    return _tar_batch_up(
        clip_dir,
        video_path,
        body_frames_dir,
        remote_run_dir,
        config,
        run=run,
        phases=phases,
        camera_motion_path=camera_motion_path,
    )


def _sync_body_outputs(
    remote_run_dir: str,
    clip_dir: Path,
    config: RemoteConfig,
    *,
    run: RunFn,
    phases: dict[str, float] | None = None,
) -> list[str]:
    transport = _validate_transport_name(config.transport)
    if transport == TRANSPORT_RSYNC:
        return _rsync_down(remote_run_dir, clip_dir, config, run=run)
    return _tar_batch_down(remote_run_dir, clip_dir, config, run=run, phases=phases)


def _tar_batch_up(
    clip_dir: Path,
    video_path: Path,
    body_frames_dir: str | Path | None,
    remote_run_dir: str,
    config: RemoteConfig,
    *,
    run: RunFn,
    phases: dict[str, float] | None = None,
    camera_motion_path: str | Path | None = None,
) -> list[str]:
    phases = phases if phases is not None else {}
    selection = _collect_body_inputs(
        clip_dir,
        video_path,
        body_frames_dir,
        camera_motion_path=camera_motion_path,
    )
    synced = selection.synced_names
    if not synced:
        return []

    with tempfile.TemporaryDirectory(prefix="remote_body_tar_up_") as tmp:
        archive_path = Path(tmp) / "body_inputs.tar.gz"
        create_started = time.monotonic()
        with tarfile.open(archive_path, "w:gz", dereference=True) as archive:
            for remote_name, source_path in selection.single_files.items():
                _add_path_to_tar(archive, source_path, remote_name)
            for remote_name, source_path in selection.directories.items():
                _add_path_to_tar(archive, source_path, remote_name)
        phases["tar_create_upload_archive_s"] = time.monotonic() - create_started

        remote_archive_path = f"{remote_run_dir}/remote_body_inputs.tar.gz"
        scp_started = time.monotonic()
        _run_transport_command(
            [*config.scp_base(), str(archive_path), f"{config.host}:{remote_archive_path}"],
            None,
            config=config,
            run=run,
            operation="tar-batch upload archive",
        )
        phases["tar_upload_scp_s"] = time.monotonic() - scp_started

    untar_started = time.monotonic()
    remote_cmd = (
        f"COPYFILE_DISABLE=1 tar -xzf {shlex.quote(remote_archive_path)} -C {shlex.quote(remote_run_dir)} "
        f"&& rm -f {shlex.quote(remote_archive_path)}"
    )
    _run_transport_command(
        [*config.ssh_base(), remote_cmd],
        config.connect_timeout_s + 60,
        config=config,
        run=run,
        operation="tar-batch remote untar",
    )
    phases["tar_remote_untar_s"] = time.monotonic() - untar_started
    return synced


def _tar_batch_down(
    remote_run_dir: str,
    clip_dir: Path,
    config: RemoteConfig,
    *,
    run: RunFn,
    phases: dict[str, float] | None = None,
) -> list[str]:
    phases = phases if phases is not None else {}
    clip_dir.mkdir(parents=True, exist_ok=True)
    requested_files = list(_download_artifacts_for_config(config))
    existing_files = _remote_existing_output_files(remote_run_dir, requested_files, config, run=run)
    existing_dirs = _remote_existing_output_dirs(remote_run_dir, BODY_OUTPUT_DIRS_DEFAULT, config, run=run)
    entries = [*existing_files, *(dirname.rstrip("/") for dirname in existing_dirs)]
    if not entries:
        return []

    remote_archive_path = f"{remote_run_dir}/remote_body_outputs.tar.gz"
    entry_args = " ".join(shlex.quote(_safe_tar_member_name(entry)) for entry in entries)
    pack_started = time.monotonic()
    remote_pack_cmd = (
        f"cd {shlex.quote(remote_run_dir)} && "
        f"rm -f {shlex.quote(remote_archive_path)} && "
        f"COPYFILE_DISABLE=1 tar -czf {shlex.quote(remote_archive_path)} {entry_args}"
    )
    _run_transport_command(
        [*config.ssh_base(), remote_pack_cmd],
        config.connect_timeout_s + 300,
        config=config,
        run=run,
        operation="tar-batch remote output pack",
    )
    phases["tar_remote_pack_s"] = time.monotonic() - pack_started

    with tempfile.TemporaryDirectory(prefix="remote_body_tar_down_") as tmp:
        archive_path = Path(tmp) / "body_outputs.tar.gz"
        download_started = time.monotonic()
        _run_transport_command(
            [*config.scp_base(), f"{config.host}:{remote_archive_path}", str(archive_path)],
            None,
            config=config,
            run=run,
            operation="tar-batch download archive",
        )
        phases["tar_download_scp_s"] = time.monotonic() - download_started

        extract_started = time.monotonic()
        _safe_extract_tar_gz(archive_path, clip_dir)
        phases["tar_extract_outputs_s"] = time.monotonic() - extract_started

    cleanup_cmd = f"rm -f {shlex.quote(remote_archive_path)}"
    cleanup_result = run([*config.ssh_base(), cleanup_cmd], config.connect_timeout_s + 10)
    if cleanup_result.returncode != 0:
        # Cleanup is best effort; the tar already downloaded and extracted.
        pass

    synced: list[str] = []
    for name in requested_files:
        if name in existing_files and (clip_dir / name).is_file():
            synced.append(name)
    for dirname in existing_dirs:
        if (clip_dir / dirname.rstrip("/")).is_dir():
            synced.append(dirname)
    return synced


def _rsync_up(
    clip_dir: Path,
    video_path: Path,
    body_frames_dir: str | Path | None,
    remote_run_dir: str,
    config: RemoteConfig,
    *,
    run: RunFn,
    camera_motion_path: str | Path | None = None,
) -> list[str]:
    if not (clip_dir / "tracks.json").is_file():
        raise RemoteBodyDispatchError("refusing remote BODY dispatch without a local tracks.json to sync (nothing to run body on)")

    synced: list[str] = []
    rsync_ssh = config.rsync_ssh_command()
    single_file_inputs: dict[str, Path] = {}

    if video_path.is_file():
        single_file_inputs["source.mp4"] = video_path

    explicit_camera_motion = Path(camera_motion_path) if camera_motion_path is not None else None
    for name in BODY_INPUT_ARTIFACTS:
        if name == CAMERA_MOTION_ARTIFACT and explicit_camera_motion is not None and explicit_camera_motion.is_file():
            continue
        local_path = clip_dir / name
        if not local_path.is_file():
            continue
        single_file_inputs[name] = local_path

    if explicit_camera_motion is not None and explicit_camera_motion.is_file():
        canonical_clip_camera_motion = clip_dir / CAMERA_MOTION_ARTIFACT
        if not canonical_clip_camera_motion.is_file() or explicit_camera_motion.resolve() != canonical_clip_camera_motion.resolve():
            single_file_inputs[CAMERA_MOTION_ARTIFACT] = explicit_camera_motion

    if single_file_inputs:
        synced.extend(
            _rsync_single_file_batch_up(
                single_file_inputs,
                remote_run_dir=remote_run_dir,
                config=config,
                rsync_ssh=rsync_ssh,
                run=run,
            )
        )

    frames_dir = Path(body_frames_dir) if body_frames_dir is not None else clip_dir / "body_frames"
    if frames_dir.is_dir() and any(frames_dir.iterdir()):
        _run_transport_command(
            ["rsync", "-az", "-e", rsync_ssh, f"{frames_dir}/", f"{config.host}:{remote_run_dir}/body_frames/"],
            None,
            config=config,
            run=run,
            operation="rsync of body_frames/",
        )
        synced.append("body_frames/")

    for dirname in SAM3D_MASK_INPUT_DIRS:
        mask_dir = clip_dir / dirname
        if not mask_dir.is_dir() or not any(mask_dir.iterdir()):
            continue
        _run_transport_command(
            ["rsync", "-az", "-e", rsync_ssh, f"{mask_dir}/", f"{config.host}:{remote_run_dir}/{dirname}/"],
            None,
            config=config,
            run=run,
            operation=f"rsync of {dirname}/",
        )
        synced.append(f"{dirname}/")

    return synced


def _rsync_down(remote_run_dir: str, clip_dir: Path, config: RemoteConfig, *, run: RunFn) -> list[str]:
    synced: list[str] = []
    rsync_ssh = config.rsync_ssh_command()
    clip_dir.mkdir(parents=True, exist_ok=True)

    requested_files = list(_download_artifacts_for_config(config))
    existing_files = _remote_existing_output_files(remote_run_dir, requested_files, config, run=run)
    if existing_files:
        with tempfile.TemporaryDirectory(prefix="remote_body_rsync_down_") as tmp:
            files_from = Path(tmp) / "outputs.txt"
            files_from.write_text("".join(f"{name}\n" for name in existing_files), encoding="utf-8")
            result = _run_transport_command(
                [
                    "rsync",
                    "-az",
                    "--files-from",
                    str(files_from),
                    "-e",
                    rsync_ssh,
                    f"{config.host}:{remote_run_dir}/",
                    f"{clip_dir}/",
                ],
                None,
                config=config,
                run=run,
                operation="rsync of BODY output batch",
                tolerated_failure=lambda failed: _rsync_missing_only(failed.stderr),
            )
            if result.returncode != 0 and not _rsync_missing_only(result.stderr):
                raise RemoteBodyDispatchError(f"rsync of BODY output batch failed: {result.stderr.strip()}")
        for name in requested_files:
            if name in existing_files and (clip_dir / name).is_file():
                synced.append(name)
    for dirname in BODY_OUTPUT_DIRS_DEFAULT:
        local_dir = clip_dir / dirname.rstrip("/")
        try:
            result = _run_transport_command(
                ["rsync", "-az", "-e", rsync_ssh, f"{config.host}:{remote_run_dir}/{dirname}", str(local_dir)],
                None,
                config=config,
                run=run,
                operation=f"rsync of {dirname}",
            )
        except RemoteBodyDispatchError:
            continue
        if result.returncode == 0 and local_dir.is_dir():
            synced.append(dirname)
    return synced


def _rsync_single_file_batch_up(
    files_by_remote_name: Mapping[str, Path],
    *,
    remote_run_dir: str,
    config: RemoteConfig,
    rsync_ssh: str,
    run: RunFn,
) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="remote_body_rsync_up_") as tmp:
        staging_dir = Path(tmp) / "files"
        staging_dir.mkdir()
        ordered_names: list[str] = []
        for remote_name, source_path in files_by_remote_name.items():
            if "/" in remote_name or remote_name in {"", ".", ".."}:
                raise RemoteBodyDispatchError(f"refusing unsafe rsync batch filename {remote_name!r}")
            staged_path = staging_dir / remote_name
            staged_path.symlink_to(source_path.resolve())
            ordered_names.append(remote_name)
        files_from = Path(tmp) / "inputs.txt"
        files_from.write_text("".join(f"{name}\n" for name in ordered_names), encoding="utf-8")
        result = _run_transport_command(
            [
                "rsync",
                "-azL",
                "--files-from",
                str(files_from),
                "-e",
                rsync_ssh,
                f"{staging_dir}/",
                f"{config.host}:{remote_run_dir}/",
            ],
            None,
            config=config,
            run=run,
            operation="rsync of BODY input file batch",
        )
    return ordered_names


def _remote_existing_output_files(
    remote_run_dir: str,
    requested_files: Sequence[str],
    config: RemoteConfig,
    *,
    run: RunFn,
) -> list[str]:
    if not requested_files:
        return []
    q = shlex.quote
    names = [str(name) for name in requested_files]
    for name in names:
        if "/" in name or name in {"", ".", ".."}:
            raise RemoteBodyDispatchError(f"refusing unsafe remote output filename {name!r}")
    name_args = " ".join(q(name) for name in names)
    command = (
        f"cd {q(remote_run_dir)} && "
        f"for f in {name_args}; do if [ -f \"$f\" ]; then printf '%s\\n' \"$f\"; fi; done "
        "# BODY_OUTPUT_FILE_LIST"
    )
    result = run([*config.ssh_base(), command], config.connect_timeout_s + 10)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RemoteBodyDispatchError(f"remote BODY output listing failed: {detail}")
    requested = set(names)
    existing = [line.strip() for line in result.stdout.splitlines() if line.strip() in requested]
    return [name for name in names if name in set(existing)]


def _remote_existing_output_dirs(
    remote_run_dir: str,
    requested_dirs: Sequence[str],
    config: RemoteConfig,
    *,
    run: RunFn,
) -> list[str]:
    if not requested_dirs:
        return []
    q = shlex.quote
    names = [str(name).rstrip("/") for name in requested_dirs]
    for name in names:
        _safe_tar_member_name(name)
        if "/" in name:
            raise RemoteBodyDispatchError(f"refusing unsafe remote output directory {name!r}")
    name_args = " ".join(q(name) for name in names)
    command = (
        f"cd {q(remote_run_dir)} && "
        f"for d in {name_args}; do if [ -d \"$d\" ]; then printf '%s/\\n' \"$d\"; fi; done "
        "# BODY_OUTPUT_DIR_LIST"
    )
    result = run([*config.ssh_base(), command], config.connect_timeout_s + 10)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RemoteBodyDispatchError(f"remote BODY output directory listing failed: {detail}")
    requested = {f"{name}/" for name in names}
    existing = [line.strip() for line in result.stdout.splitlines() if line.strip() in requested]
    return [f"{name}/" for name in names if f"{name}/" in set(existing)]


def _rsync_missing_only(stderr: str) -> bool:
    text = (stderr or "").lower()
    if not text:
        return False
    missing_markers = ("no such file", "vanished file", "link_stat", "(l)stat")
    return any(marker in text for marker in missing_markers) and not any(
        marker in text for marker in ("permission denied", "connection refused", "connection reset", "ssh:")
    )


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
    parser.add_argument("--camera-motion", type=Path, default=None, help="Optional camera_motion.json to sync as camera_motion.json in the remote BODY working dir.")
    parser.add_argument("--host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--ssh-key", default=DEFAULT_SSH_KEY)
    parser.add_argument("--repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--fast-sam-python", default=DEFAULT_REMOTE_FAST_SAM_PYTHON)
    parser.add_argument("--fast-sam-root", default=DEFAULT_REMOTE_FAST_SAM_ROOT)
    parser.add_argument(
        "--transport",
        choices=TRANSPORT_CHOICES,
        default=RemoteConfig().transport,
        help="Remote BODY transport: tar_batch is the hardened default; rsync is the selectable GNU-rsync fallback.",
    )
    parser.add_argument("--transport-retry-max-attempts", type=int, default=RemoteConfig().transport_retry_max_attempts)
    parser.add_argument("--transport-retry-backoff-s", type=float, default=RemoteConfig().transport_retry_backoff_s)
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
    parser.add_argument(
        "--fetch-body-monoliths",
        action="store_true",
        help="Also download smpl_motion.json and body_mesh.json. Default skips them for faster BODY round trips.",
    )
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
        transport=args.transport,
        transport_retry_max_attempts=args.transport_retry_max_attempts,
        transport_retry_backoff_s=args.transport_retry_backoff_s,
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
        fetch_body_monoliths=bool(args.fetch_body_monoliths),
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
            camera_motion_path=args.camera_motion,
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
