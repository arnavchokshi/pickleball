#!/usr/bin/env python3
"""Operator CLI: lifecycle management for a persistent warm SAM-3D-Body worker.

warm_body_worker_20260728. Speed lever #1 in
``runs/court_skeleton_runtime_20260725/REPORT.md`` ("BODY speed truth"):
measured model-load (~24.9s) plus torch.compile warmup (~33.5s) cost about 58s
of pure cold-start waste on *every* remote BODY dispatch, dominated by
one-time bring-up rather than steady-state inference (~5.5-5.8s/clip).

This module is the SSH-orchestrated lifecycle tool for
``scripts/racketsport/sam3dbody_persistent_worker.py`` (already committed by
the body_overhead_20260712 lane: it loads the checkpoint once, warms the
compile buckets once, then serves batch jobs over a local Unix socket with
fingerprint validation, a CUDA-context health canary, a consecutive-crash
kill switch, and idle self-teardown). That lane proved the mechanism correct
but scored it default-off/opt-in only, with no operator-facing way to start,
health-check, or stop a worker from a laptop client, and no story for how the
worker cooperates with the shared GPU lock across its own long-lived serving
window. This module is that missing piece; ``scripts/racketsport/
remote_body_dispatch.py``'s ``--warm-worker`` flag is the per-job client half
of the same feature (health-probes a worker started here, routes a job to it
when healthy, falls back loudly to the unchanged cold path otherwise).

Subcommands
-----------
``start``
    Runs ONE real, ordinary cold BODY dispatch for --clip (identical to
    running remote_body_dispatch.py directly -- same artifacts, same
    version-stamp verification, same shared-lock-guarded remote command).
    This is deliberate, not incidental: it is the simplest way to obtain a
    real ``batch_requests-*.json`` payload with the exact request shape a
    later warm job for this clip will send, so the persistent worker's
    conservative fingerprint check (worker-boot config + the FULL
    clip_intrinsics matrix, per body_overhead_20260712's design) has
    something genuine to match against instead of a synthetic stand-in.
    After that cold run completes, ``start`` locates the request payload it
    produced on the remote host, resolves the SAM-3D-Body checkpoint
    directory through the same models-manifest verification the real
    pipeline uses, and launches
    ``sam3dbody_persistent_worker.py serve`` in the background over SSH,
    itself wrapped in the shared ``scripts/gpu-eval-run.sh`` lock so it holds
    that lock continuously for its entire serving lifetime. That is the
    lock-cooperation contract: any cold BODY dispatch that does NOT go
    through the warm worker still wraps with the same lock script exactly as
    before, so it correctly queues behind the worker's held lock instead of
    racing its already-open CUDA context under ``nvidia-smi -c
    EXCLUSIVE_PROCESS``. A per-job warm dispatch (remote_body_dispatch.py
    --warm-worker) in turn skips that same lock wrap for its own brief
    command, because re-wrapping it would self-block against the lock the
    worker already holds.

    ``start`` writes a typed remote manifest sidecar
    (``<socket-path>.manifest.json``) recording the git commit the worker was
    started from, the clip it was bootstrapped for, its process id, and its
    bootstrap timing -- this is what ``status``/``--warm-worker`` health
    probes and ``stop`` all read back.

``status``
    Runs the identical health probe remote_body_dispatch.py's --warm-worker
    flag runs before routing a job: manifest present, socket reachable, code
    stamp still matches local HEAD, clip matches. Exits 0 when healthy, 1
    otherwise, always printing the full typed result.

``stop``
    Reads the remote manifest for the recorded pid, sends SIGTERM over SSH,
    confirms the socket stops accepting connections, and removes the
    manifest/ready sidecar files. Absent-worker is reported, not an error
    (idempotent).

Everything here is a thin SSH/JSON orchestration layer over already-tested
code (``dispatch_body_stage``, ``sam3dbody_persistent_worker.py``); it does
not reimplement BODY inference or change any BODY artifact contract.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport import remote_body_dispatch as rbd  # noqa: E402

RunFn = rbd.RunFn
SleepFn = Callable[[float], None]

DEFAULT_IDLE_TIMEOUT_S = 1200.0
DEFAULT_MAX_CONSECUTIVE_JOB_CRASHES = 2
DEFAULT_READY_TIMEOUT_S = 240.0
DEFAULT_POLL_INTERVAL_S = 5.0
DEFAULT_STOP_TIMEOUT_S = 60.0
DEFAULT_STOP_POLL_INTERVAL_S = 3.0
WARM_WORKER_MANIFEST_ARTIFACT_TYPE = "racketsport_body_warm_worker_manifest"


class WarmWorkerLifecycleError(rbd.RemoteBodyDispatchError):
    """Raised when start/status/stop cannot complete for a real, typed reason."""


def _add_common_remote_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="", help=rbd.REMOTE_HOST_REQUIRED_MESSAGE)
    parser.add_argument("--ssh-key", default=rbd.DEFAULT_SSH_KEY)
    parser.add_argument("--repo", default=rbd.DEFAULT_REMOTE_REPO)
    parser.add_argument("--python", default=rbd.DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--fast-sam-python", default=rbd.DEFAULT_REMOTE_FAST_SAM_PYTHON)
    parser.add_argument("--fast-sam-root", default=rbd.DEFAULT_REMOTE_FAST_SAM_ROOT)
    parser.add_argument("--known-hosts-file", default=rbd.DEFAULT_KNOWN_HOSTS_FILE)
    parser.add_argument("--gpu-lock-script", default=rbd.DEFAULT_GPU_LOCK_SCRIPT)
    parser.add_argument("--run-root", default=rbd.DEFAULT_RUN_ROOT)
    parser.add_argument("--connect-timeout-s", type=int, default=rbd.DEFAULT_SSH_CONNECT_TIMEOUT_S)


def _remote_config_from_common_args(args: argparse.Namespace, **overrides: Any) -> rbd.RemoteConfig:
    fields: dict[str, Any] = dict(
        host=args.host,
        ssh_key=args.ssh_key,
        repo=args.repo,
        python=args.python,
        fast_sam_python=args.fast_sam_python,
        fast_sam_root=args.fast_sam_root,
        known_hosts_file=args.known_hosts_file,
        gpu_lock_script=args.gpu_lock_script,
        run_root=args.run_root,
        connect_timeout_s=args.connect_timeout_s,
    )
    fields.update(overrides)
    return rbd.RemoteConfig(**fields)


def _build_start_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    start = sub.add_parser(
        "start",
        help="Run one cold BODY dispatch, then keep a persistent worker warm from its real request payload.",
    )
    _add_common_remote_arguments(start)
    start.add_argument("--clip", required=True)
    start.add_argument("--clip-dir", type=Path, required=True, help="Local directory with tracks.json/etc.")
    start.add_argument("--video", type=Path, required=True)
    start.add_argument("--body-frames-dir", type=Path, default=None)
    start.add_argument("--camera-motion", type=Path, default=None)
    start.add_argument("--transport", choices=rbd.TRANSPORT_CHOICES, default=rbd.RemoteConfig().transport)
    start.add_argument("--lock-wait-timeout-s", type=int, default=rbd.DEFAULT_LOCK_WAIT_TIMEOUT_S)
    start.add_argument("--command-timeout-s", type=int, default=rbd.DEFAULT_COMMAND_TIMEOUT_S)
    start.add_argument("--pipeline-preset", choices=("full", "court_skeletons"), default="full")
    start.add_argument("--max-frames", type=int, default=None)
    start.add_argument("--max-players", type=int, default=4)
    start.add_argument("--allow-dirty", action="store_true")
    start.add_argument("--sam3d-body-input-size-px", type=int, choices=(384, 448, 512), default=384)
    start.add_argument("--sam3d-crop-bucket-sizes", default="8,16")
    start.add_argument("--sam3d-compile-warmup-buckets", default="8,16")
    start.add_argument("--sam3d-compile-warmup-passes", type=int, default=2)
    start.add_argument("--no-sam3d-torch-compile", action="store_true")
    start.add_argument("--body-detector-name", default=rbd.DEFAULT_BODY_DETECTOR_NAME)
    start.add_argument("--body-fov-name", default=rbd.DEFAULT_BODY_FOV_NAME)
    start.add_argument(
        "--socket-path",
        default="",
        help="Remote Unix socket path. Default: derived from --repo/--run-root/--clip (default_warm_worker_socket_path).",
    )
    start.add_argument("--idle-timeout-s", type=float, default=DEFAULT_IDLE_TIMEOUT_S)
    start.add_argument("--max-consecutive-job-crashes", type=int, default=DEFAULT_MAX_CONSECUTIVE_JOB_CRASHES)
    start.add_argument("--ready-timeout-s", type=float, default=DEFAULT_READY_TIMEOUT_S)
    start.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    start.add_argument(
        "--replace",
        action="store_true",
        help="Stop an already-running worker at the resolved socket path first instead of refusing to start.",
    )
    start.add_argument("--out", type=Path, default=None, help="Optional local path to also write the manifest JSON to.")


def _build_status_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    status = sub.add_parser(
        "status", help="Health-check a warm worker (manifest + socket reachability + code-stamp match)."
    )
    _add_common_remote_arguments(status)
    status.add_argument("--clip", required=True)
    status.add_argument("--socket-path", default="")
    status.add_argument("--health-timeout-s", type=float, default=rbd.RemoteConfig().warm_worker_health_timeout_s)


def _build_stop_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    stop = sub.add_parser(
        "stop", help="Stop a running warm worker (SIGTERM by recorded pid) and confirm it exited."
    )
    _add_common_remote_arguments(stop)
    stop.add_argument("--clip", default=None)
    stop.add_argument("--socket-path", default="")
    stop.add_argument("--timeout-s", type=float, default=DEFAULT_STOP_TIMEOUT_S)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Lifecycle CLI for a persistent warm SAM-3D-Body worker (warm_body_worker_20260728), "
            "default-off companion to remote_body_dispatch.py's --warm-worker flag."
        )
    )
    sub = parser.add_subparsers(dest="mode", required=True)
    _build_start_parser(sub)
    _build_status_parser(sub)
    _build_stop_parser(sub)
    return parser


def _resolve_checkpoint_dir(
    config: rbd.RemoteConfig,
    *,
    detector_name: str,
    fov_name: str,
    manifest_path: str,
    run: RunFn,
) -> str:
    """Resolve the SAM-3D-Body checkpoint dir the same way the real pipeline does.

    Runs ``threed.racketsport.hmr_deep.verify_fast_sam_manifest_assets`` on
    the remote host (the exact function ``BodyStageRunner`` calls) rather
    than hardcoding a path here, so this stays correct if the models
    manifest ever changes.
    """

    script = (
        "import json\n"
        "from threed.racketsport.hmr_deep import verify_fast_sam_manifest_assets, fast_sam_required_model_ids\n"
        f"required = fast_sam_required_model_ids(detector_name={detector_name!r}, fov_name={fov_name!r})\n"
        f"assets = verify_fast_sam_manifest_assets({manifest_path!r}, required_model_ids=required)\n"
        "print(json.dumps({'checkpoint_dir': str(assets['fast_sam_3d_body_dinov3'].path.parent)}))\n"
    )
    command = f"cd {shlex.quote(config.repo)} && {shlex.quote(config.python)} -c {shlex.quote(script)}"
    result = run([*config.ssh_base(), command], config.connect_timeout_s + 60)
    if result.returncode != 0:
        raise WarmWorkerLifecycleError(
            f"could not resolve the SAM-3D-Body checkpoint dir on {config.host} via the models manifest "
            f"(exit {result.returncode}): {(result.stderr or result.stdout or '').strip()[-1000:]}"
        )
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1]) if lines else {}
        return str(payload["checkpoint_dir"])
    except (ValueError, IndexError, KeyError) as exc:
        raise WarmWorkerLifecycleError(
            f"unexpected output resolving checkpoint dir on {config.host}: {type(exc).__name__}: {exc}; "
            f"stdout={(result.stdout or '')[-500:]!r}"
        ) from exc


def _find_bootstrap_requests(config: rbd.RemoteConfig, remote_run_dir: str, *, run: RunFn) -> str:
    """Locate the batch_requests-*.json a just-completed cold dispatch produced.

    threed.racketsport.orchestrator writes exactly one
    ``batch_requests-<uuid>.json`` per BODY subprocess invocation directly
    under the run's work_dir (== remote_run_dir here); this is that file's
    real, on-disk request payload -- not a synthetic stand-in -- so the
    persistent worker's fingerprint will genuinely match later same-clip jobs.
    """

    command = f"ls -1 {shlex.quote(remote_run_dir)}/batch_requests-*.json 2>/dev/null | sort | tail -n 1"
    result = run([*config.ssh_base(), command], config.connect_timeout_s + 20)
    stdout = (result.stdout or "").strip()
    path = stdout.splitlines()[-1].strip() if stdout else ""
    if result.returncode != 0 or not path:
        raise WarmWorkerLifecycleError(
            f"no batch_requests-*.json found under {remote_run_dir} on {config.host}; the inline cold "
            "bootstrap dispatch should have produced one -- this indicates the BODY stage did not reach "
            "the SAM3D request-building step (check remote_body_stdout.log in --clip-dir)."
        )
    return path


def _probe_manifest_and_socket(
    config: rbd.RemoteConfig, socket_path: str, *, connect_timeout_s: float, run: RunFn
) -> dict[str, Any]:
    """Lighter-weight probe than probe_warm_worker_health: no clip/git-sha comparison.

    Used by start (to decide whether an existing worker needs --replace) and
    stop (which only cares whether a process/socket is present, not whether
    it is safe to route a job to).
    """

    manifest_path = rbd._warm_worker_manifest_path(socket_path)
    script = rbd._warm_worker_probe_script(manifest_path=manifest_path, connect_timeout_s=connect_timeout_s)
    command = f"{shlex.quote(config.python)} -c {shlex.quote(script)}"
    result = run([*config.ssh_base(), command], config.connect_timeout_s + connect_timeout_s + 10)
    if result.returncode != 0:
        raise WarmWorkerLifecycleError(
            f"probe SSH failed on {config.host} (exit {result.returncode}): {(result.stderr or '').strip()[-500:]}"
        )
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if not lines:
        return {"manifest_found": False}
    try:
        payload = json.loads(lines[-1])
    except ValueError as exc:
        raise WarmWorkerLifecycleError(
            f"non-JSON probe output on {config.host}: {(result.stdout or '')[-300:]!r}"
        ) from exc
    return payload if isinstance(payload, dict) else {"manifest_found": False}


def _poll_for_ready(
    config: rbd.RemoteConfig,
    *,
    ready_path: str,
    timeout_s: float,
    poll_interval_s: float,
    log_path: str,
    run: RunFn,
    sleep: SleepFn = time.sleep,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    check_cmd = f"cat {shlex.quote(ready_path)} 2>/dev/null"
    while True:
        result = run([*config.ssh_base(), check_cmd], config.connect_timeout_s + 20)
        text = (result.stdout or "").strip()
        if text:
            try:
                return json.loads(text)
            except ValueError:
                pass  # ready file can be mid-write; retry
        if time.monotonic() >= deadline:
            raise WarmWorkerLifecycleError(
                f"warm worker did not become ready within {timeout_s}s on {config.host}; check "
                f"{log_path} on the remote host for the real failure (model load, torch.compile, or a "
                "long shared-GPU-lock wait)"
            )
        sleep(poll_interval_s)


def start_worker(
    args: argparse.Namespace, *, run: RunFn = rbd._run, sleep: SleepFn = time.sleep
) -> dict[str, Any]:
    clip = rbd._validate_clip_id(args.clip)
    config = _remote_config_from_common_args(
        args,
        transport=args.transport,
        lock_wait_timeout_s=args.lock_wait_timeout_s,
        command_timeout_s=args.command_timeout_s,
        sam3d_body_input_size_px=args.sam3d_body_input_size_px,
        sam3d_crop_bucket_sizes=rbd._parse_int_tuple(
            args.sam3d_crop_bucket_sizes, flag_name="--sam3d-crop-bucket-sizes"
        ),
        sam3d_compile_warmup_buckets=rbd._parse_int_tuple(
            args.sam3d_compile_warmup_buckets, flag_name="--sam3d-compile-warmup-buckets"
        ),
        sam3d_compile_warmup_passes=args.sam3d_compile_warmup_passes,
        sam3d_torch_compile=not args.no_sam3d_torch_compile,
        body_detector_name=args.body_detector_name,
        body_fov_name=args.body_fov_name,
        # The bootstrap dispatch below must always be a genuine cold run
        # (there is no worker to route it to yet); warm_worker is not exposed
        # as a start.py flag at all, this just documents that it stays False.
        warm_worker=False,
    )
    rbd._require_remote_host(config.host)

    socket_path = args.socket_path or rbd.default_warm_worker_socket_path(config, clip)
    manifest_path = rbd._warm_worker_manifest_path(socket_path)
    ready_path = f"{socket_path}.ready.json"
    log_path = f"{socket_path}.log"

    existing = _probe_manifest_and_socket(
        config, socket_path, connect_timeout_s=config.warm_worker_health_timeout_s, run=run
    )
    if existing.get("manifest_found") and existing.get("socket_reachable"):
        if not args.replace:
            raise WarmWorkerLifecycleError(
                f"a warm worker is already running at {socket_path} on {config.host}; pass --replace to "
                "stop it first, or choose a different --socket-path"
            )
        stop_result = stop_worker(config, clip=None, socket_path=socket_path, timeout_s=60.0, run=run, sleep=sleep)
        print(
            f"[body_warm_worker] --replace: stopped the existing worker first: {json.dumps(stop_result)}",
            file=sys.stderr,
        )

    print(
        f"[body_warm_worker] running one real cold BODY dispatch for clip {clip!r} on {config.host} to "
        "produce a real batch_requests-*.json payload to bootstrap the worker from (this is the same "
        "dispatch remote_body_dispatch.py would run; it is not skipped or faked)...",
        file=sys.stderr,
    )
    bootstrap_started = time.monotonic()
    cold_result = rbd.dispatch_body_stage(
        clip=clip,
        clip_dir=args.clip_dir,
        video_path=args.video,
        body_frames_dir=args.body_frames_dir,
        camera_motion_path=args.camera_motion,
        config=config,
        max_frames=args.max_frames,
        max_players=args.max_players,
        pipeline_preset=args.pipeline_preset,
        allow_dirty=bool(args.allow_dirty),
        run=run,
    )
    bootstrap_cold_dispatch_wall_s = time.monotonic() - bootstrap_started

    bootstrap_requests_path = _find_bootstrap_requests(config, cold_result.remote_run_dir, run=run)
    checkpoint_dir = _resolve_checkpoint_dir(
        config,
        detector_name=config.body_detector_name,
        fov_name=config.body_fov_name,
        manifest_path=f"{config.repo}/models/MANIFEST.json",
        run=run,
    )

    socket_parent = Path(socket_path).parent.as_posix()
    mkdir_result = run([*config.ssh_base(), f"mkdir -p {shlex.quote(socket_parent)}"], config.connect_timeout_s + 20)
    if mkdir_result.returncode != 0:
        raise WarmWorkerLifecycleError(
            f"could not create the warm-worker socket directory {socket_parent} on {config.host}: "
            f"{(mkdir_result.stderr or '').strip()[-500:]}"
        )

    serve_cmd = " ".join(
        [
            shlex.quote(config.python),
            "scripts/racketsport/sam3dbody_persistent_worker.py",
            "serve",
            "--fast-sam-repo",
            shlex.quote(config.fast_sam_root),
            "--checkpoint-dir",
            shlex.quote(checkpoint_dir),
            "--detector-name",
            shlex.quote(config.body_detector_name or ""),
            "--fov-name",
            shlex.quote(config.body_fov_name or ""),
            "--bootstrap-requests",
            shlex.quote(bootstrap_requests_path),
            "--socket-path",
            shlex.quote(socket_path),
            "--ready-path",
            shlex.quote(ready_path),
            "--idle-timeout-s",
            str(args.idle_timeout_s),
            "--max-consecutive-job-crashes",
            str(args.max_consecutive_job_crashes),
        ]
    )
    # Launched detached (setsid + nohup + redirected stdio) so it survives
    # this SSH session closing, and wrapped in config.gpu_lock_script so it
    # holds the shared slot lease for its entire serving lifetime -- see the
    # module docstring's lock-cooperation contract.
    start_cmd = (
        f"cd {shlex.quote(config.repo)} && "
        f"rm -f {shlex.quote(ready_path)} && "
        f"nohup env GPU_LOCK_TIMEOUT_S={int(args.lock_wait_timeout_s)} "
        f"FAST_SAM_PYTHON={shlex.quote(config.fast_sam_python)} FAST_SAM_ROOT={shlex.quote(config.fast_sam_root)} "
        f"setsid {shlex.quote(config.gpu_lock_script)} {serve_cmd} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo $!"
    )
    start_result = run([*config.ssh_base(), start_cmd], config.connect_timeout_s + 20)
    if start_result.returncode != 0:
        raise WarmWorkerLifecycleError(
            f"failed to launch the warm worker on {config.host} (exit {start_result.returncode}): "
            f"{(start_result.stderr or '').strip()[-1000:]}"
        )
    stdout = (start_result.stdout or "").strip()
    shell_pid = stdout.splitlines()[-1].strip() if stdout else ""

    ready_payload = _poll_for_ready(
        config,
        ready_path=ready_path,
        timeout_s=args.ready_timeout_s,
        poll_interval_s=args.poll_interval_s,
        log_path=log_path,
        run=run,
        sleep=sleep,
    )

    local_git_head_sha = rbd._git_head_sha(rbd.ROOT)
    manifest = {
        "schema_version": 1,
        "artifact_type": WARM_WORKER_MANIFEST_ARTIFACT_TYPE,
        "clip": clip,
        "host": config.host,
        "socket_path": socket_path,
        "manifest_path": manifest_path,
        "ready_path": ready_path,
        "log_path": log_path,
        "git_head_sha": local_git_head_sha,
        "pid": ready_payload.get("pid") or shell_pid,
        "shell_wrapper_pid": shell_pid,
        "started_at_utc": rbd._utc_now_iso(),
        "started_under_gpu_lock": True,
        "gpu_lock_script": config.gpu_lock_script,
        "checkpoint_dir": checkpoint_dir,
        "detector_name": config.body_detector_name,
        "fov_name": config.body_fov_name,
        "bootstrap_remote_run_dir": cold_result.remote_run_dir,
        "bootstrap_requests_path": bootstrap_requests_path,
        "bootstrap_cold_dispatch_wall_s": round(bootstrap_cold_dispatch_wall_s, 3),
        "bootstrap_timing_summary": ready_payload.get("bootstrap_timing_summary"),
        "fingerprint": ready_payload.get("fingerprint"),
        "idle_timeout_s": args.idle_timeout_s,
        "max_consecutive_job_crashes": args.max_consecutive_job_crashes,
    }
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    write_manifest_cmd = (
        f"cat > {shlex.quote(manifest_path)} <<'BODY_WARM_WORKER_MANIFEST_EOF'\n"
        f"{manifest_json}\n"
        "BODY_WARM_WORKER_MANIFEST_EOF\n"
    )
    write_result = run([*config.ssh_base(), write_manifest_cmd], config.connect_timeout_s + 20)
    if write_result.returncode != 0:
        raise WarmWorkerLifecycleError(
            f"worker started (pid={manifest['pid']}) but failed to write its manifest at {manifest_path} "
            f"on {config.host}: {(write_result.stderr or '').strip()[-500:]}; the worker is running but "
            "will not be discoverable by status/stop/--warm-worker until this is fixed by hand"
        )
    return manifest


def stop_worker(
    config: rbd.RemoteConfig,
    *,
    clip: str | None,
    socket_path: str | None,
    timeout_s: float,
    run: RunFn = rbd._run,
    sleep: SleepFn = time.sleep,
    poll_interval_s: float = DEFAULT_STOP_POLL_INTERVAL_S,
) -> dict[str, Any]:
    if not socket_path and not clip:
        raise WarmWorkerLifecycleError("stop requires --clip or --socket-path")
    resolved_socket_path = socket_path or rbd.default_warm_worker_socket_path(config, rbd._validate_clip_id(clip or ""))

    probe = _probe_manifest_and_socket(config, resolved_socket_path, connect_timeout_s=10.0, run=run)
    if not probe.get("manifest_found"):
        return {
            "status": "absent",
            "socket_path": resolved_socket_path,
            "detail": "no manifest found on the remote host; nothing to stop",
        }

    manifest = probe.get("manifest") or {}
    pid = manifest.get("pid")
    if not pid:
        raise WarmWorkerLifecycleError(
            f"manifest at {resolved_socket_path}{rbd.WARM_WORKER_MANIFEST_SUFFIX} on {config.host} has no recorded pid"
        )

    run([*config.ssh_base(), f"kill {int(pid)} 2>/dev/null; true"], config.connect_timeout_s + 20)

    deadline = time.monotonic() + timeout_s
    last_probe = probe
    while last_probe.get("socket_reachable") and time.monotonic() < deadline:
        sleep(poll_interval_s)
        last_probe = _probe_manifest_and_socket(config, resolved_socket_path, connect_timeout_s=5.0, run=run)

    stopped = not last_probe.get("socket_reachable")
    cleanup_cmd = (
        f"rm -f {shlex.quote(resolved_socket_path)}{rbd.WARM_WORKER_MANIFEST_SUFFIX} "
        f"{shlex.quote(resolved_socket_path)}.ready.json 2>/dev/null; true"
    )
    if stopped:
        run([*config.ssh_base(), cleanup_cmd], config.connect_timeout_s + 20)

    return {
        "status": "stopped" if stopped else "stop_timed_out",
        "socket_path": resolved_socket_path,
        "pid": pid,
        "detail": (
            "sent SIGTERM and confirmed the socket is no longer reachable"
            if stopped
            else f"sent SIGTERM but the socket was still reachable after {timeout_s}s; check the remote process by hand"
        ),
    }


def cmd_status(args: argparse.Namespace, *, run: RunFn = rbd._run) -> int:
    config = _remote_config_from_common_args(args, warm_worker_health_timeout_s=args.health_timeout_s)
    health = rbd.probe_warm_worker_health(config, clip=args.clip, socket_path=args.socket_path or None, run=run)
    print(json.dumps(health.as_dict(), indent=2, sort_keys=True))
    return 0 if health.healthy else 1


def cmd_stop(args: argparse.Namespace, *, run: RunFn = rbd._run) -> int:
    config = _remote_config_from_common_args(args)
    try:
        result = stop_worker(
            config, clip=args.clip, socket_path=args.socket_path or None, timeout_s=args.timeout_s, run=run
        )
    except WarmWorkerLifecycleError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"stopped", "absent"} else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.mode == "start":
            manifest = start_worker(args)
            if args.out is not None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(manifest, indent=2, sort_keys=True))
            return 0
        if args.mode == "status":
            return cmd_status(args)
        if args.mode == "stop":
            return cmd_stop(args)
    except (rbd.RemoteBodyDispatchError, WarmWorkerLifecycleError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    parser.error(f"unknown mode {args.mode!r}")
    return 2  # pragma: no cover - argparse.error() already raises SystemExit


if __name__ == "__main__":
    raise SystemExit(main())
