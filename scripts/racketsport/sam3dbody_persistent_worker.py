#!/usr/bin/env python3
"""Experimental persistent SAM-3D-Body worker (body_overhead_20260712, Lever A).

Scope and status
-----------------
This is a SCOPED NS-06 orchestration-overhead experiment, not a default or a
promotion. `VERIFIED=0`. Every normal invocation of
`scripts/racketsport/run_sam3dbody_batch.py` and
`scripts/racketsport/remote_body_dispatch.py` is completely unaffected unless
an operator explicitly opts in (empty `SAM3DBODY_PERSISTENT_WORKER_SOCKET` env
var / unset `RemoteConfig.sam3dbody_persistent_worker_socket`).

Why this exists
----------------
Each BODY dispatch on the committed default path spawns a fresh
`run_sam3dbody_batch.py` subprocess that: imports torch and the vendored
FastSAM-3D-Body runtime, constructs a `SAM3DBodyEstimator` from a checkpoint
directory (`model_setup_load`, ~13s measured on H100), and runs a
static-clip-intrinsics `torch.compile` warmup (`compile_warmup`, ~24-31s
measured on H100) before doing any real inference. Steady-state inference for
an entire Wolverine-sized clip (705 player-frames) is only ~5.5-5.8s, so this
one-time bring-up cost dominates repeated same-VM BODY dispatches.

This module keeps ONE already-loaded, already-compiled estimator resident in
a long-lived process and serves multiple BODY batch jobs against it over a
local Unix domain socket, so the second and later jobs in one VM session pay
none of the model-load/compile-warmup tax. It reuses
`run_sam3dbody_batch._run_batch_inference_and_write` for the actual
inference/output-writing code path, so a served job executes byte-identical
logic to the direct-subprocess path -- only model construction/compile is
skipped on repeat jobs.

Safety properties (explicit accept/kill requirements for this lever)
----------------------------------------------------------------------
- Fingerprint validation per job: every job's `optimization` + `clip_intrinsics`
  plus the worker's own fixed checkpoint/detector/code identity are hashed
  into a single fingerprint at bootstrap; a job whose own recomputed
  fingerprint does not match is REFUSED (`fingerprint_mismatch`), never
  silently processed against a resident model/config it does not match. This
  matters because upstream FastSAM modules read environment variables (e.g.
  `USE_COMPILE`) during import, so the config used at bootstrap is frozen for
  the life of the process; a job cannot retroactively change it.
- Stale GPU context prevention: before every job, a cheap CUDA canary
  (`synchronize()` + a tiny tensor round-trip) must succeed. If it raises, the
  worker marks itself unhealthy, refuses the job (`worker_unhealthy`), and
  exits rather than silently continuing to serve on a possibly-corrupted CUDA
  context.
- Consecutive-crash kill switch: any job whose inference/output-writing raises,
  or any protocol-level failure, increments a crash counter; reaching
  `--max-consecutive-job-crashes` (default 2) makes the worker exit after
  responding to the failing job. A clean `status="ok"` resets the counter.
- Idle self-teardown: with no job activity for `--idle-timeout-s`, the worker
  exits and removes its own socket file, so an orphaned worker cannot hold a
  GPU indefinitely if the calling harness dies first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.racketsport.run_sam3dbody_batch import (  # noqa: E402
    SAM3D_BATCH_TIMING_STDOUT_MARKER,
    _apply_bucket_size_override,
    _configure_runtime_environment,
    _default_chunk_dir,
    _detect_mhr_correctives_active,
    _detector_name,
    _json_safe,
    _load_setup_sam_3d_body,
    _parse_batch_payload,
    _run_batch_inference_and_write,
    _runtime_path_errors,
    _sam3d_batch_timing_summary,
    _setup_estimator,
    _timing_sidecar_path,
    _TimingRecorder,
    _warmup_static_clip_intrinsics,
    _write_failure_output,
    _write_json_payload,
)
from threed.racketsport.sam3d_body_input_prep import normalize_body_input_size  # noqa: E402

WORKER_READY_ARTIFACT_TYPE = "racketsport_sam3dbody_persistent_worker_ready"
WORKER_EXIT_ARTIFACT_TYPE = "racketsport_sam3dbody_persistent_worker_exit"
_MESSAGE_LENGTH_PREFIX_BYTES = 8
_CRITICAL_CODE_FILES = (
    "scripts/racketsport/sam3dbody_persistent_worker.py",
    "scripts/racketsport/run_sam3dbody_batch.py",
    "scripts/racketsport/run_sam3dbody_frame.py",
    "scripts/racketsport/run_sam3dbody_probe.py",
    "threed/racketsport/sam3d_body_input_prep.py",
)


class PersistentWorkerError(RuntimeError):
    """Base error for persistent-worker client/protocol failures."""


class PersistentWorkerUnavailable(PersistentWorkerError):
    """The worker socket could not be reached (not started, crashed, wrong path)."""


def _hash_code_identity(*, repo_root: Path = ROOT) -> str:
    """Content hash of every file this worker's correctness depends on.

    Used as one component of the job fingerprint: if any of these files
    change (e.g. a mid-experiment code edit), the fingerprint changes and
    every subsequent job is refused rather than silently served by stale code.
    """

    hasher = hashlib.sha256()
    for rel_path in _CRITICAL_CODE_FILES:
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(b"\0")
        try:
            hasher.update((repo_root / rel_path).read_bytes())
        except OSError as exc:
            hasher.update(f"MISSING:{type(exc).__name__}:{exc}".encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def compute_config_fingerprint(
    *,
    checkpoint_dir: Path | str,
    detector_name: str,
    detector_model: str,
    fov_name: str,
    optimization: Mapping[str, Any],
    clip_intrinsics: Mapping[str, Any] | None,
    code_identity: str,
) -> str:
    """Stable fingerprint of everything a served job must match to be safe.

    Deliberately conservative: the full `clip_intrinsics` matrix is included
    (not just its shape), so this experiment only claims correctness for
    repeat jobs against the SAME clip/config the worker booted with. Serving
    a different clip/config safely is future scope, not claimed here.
    """

    structure = {
        "code_identity": code_identity,
        "checkpoint_dir": str(Path(checkpoint_dir).resolve()),
        "detector_name": str(detector_name or ""),
        "detector_model": str(detector_model or ""),
        "fov_name": str(fov_name or ""),
        "optimization": _json_safe(dict(optimization)),
        "clip_intrinsics_matrix": _json_safe(clip_intrinsics.get("matrix") if clip_intrinsics else None),
    }
    encoded = json.dumps(structure, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _send_message(sock: socket.socket, message: Mapping[str, Any]) -> None:
    body = json.dumps(_json_safe(dict(message)), separators=(",", ":")).encode("utf-8")
    header = len(body).to_bytes(_MESSAGE_LENGTH_PREFIX_BYTES, "big")
    sock.sendall(header + body)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed before the expected message was fully received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock: socket.socket) -> dict[str, Any]:
    header = _recv_exact(sock, _MESSAGE_LENGTH_PREFIX_BYTES)
    size = int.from_bytes(header, "big")
    body = _recv_exact(sock, size)
    parsed = json.loads(body.decode("utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("persistent-worker protocol message must be a JSON object")
    return parsed


def submit_job_via_client(
    *,
    socket_path: str,
    requests_path: Path,
    out_path: Path,
    chunk_dir: Path | None,
    chunk_format: str,
    no_monolithic_output: bool,
    bucket_size: int | None,
    body_input_size: int | None = None,
    connect_timeout_s: float = 10.0,
) -> int:
    """Submit one job to an already-running worker; mirrors run_sam3dbody_batch.py's CLI contract.

    Returns a process-style exit code and prints the same
    `SAM3D_BATCH_TIMING_STDOUT_MARKER` + timing JSON line and final `out_path`
    line that `run_sam3dbody_batch.py main()` prints on success, so nothing
    downstream (orchestrator.py's subprocess handling, timing attribution)
    needs to change to consume either code path.
    """

    job = {
        "op": "run_batch",
        "requests_path": str(requests_path),
        "out_path": str(out_path),
        "chunk_dir": str(chunk_dir) if chunk_dir is not None else None,
        "chunk_format": str(chunk_format or "pickle"),
        "no_monolithic_output": bool(no_monolithic_output),
        "bucket_size": bucket_size,
        "body_input_size": body_input_size,
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(connect_timeout_s)
            sock.connect(str(socket_path))
            # No client-side deadline on the job itself: the server-side GPU
            # lock / --remote-command-timeout-s already bounds the overall
            # remote command from the dispatcher side.
            sock.settimeout(None)
            _send_message(sock, job)
            response = _recv_message(sock)
    except (OSError, ConnectionError, ValueError) as exc:
        print(
            f"sam3dbody persistent worker unreachable at {socket_path!r}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    status = str(response.get("status", ""))
    if status == "ok":
        timing_summary = response.get("timing_summary")
        if isinstance(timing_summary, dict):
            print(SAM3D_BATCH_TIMING_STDOUT_MARKER + json.dumps(timing_summary, separators=(",", ":"), sort_keys=True))
        print(response.get("out_path", str(out_path)))
        return 0

    print(f"sam3dbody persistent worker job did not succeed: status={status!r} response={response}", file=sys.stderr)
    exit_code = response.get("exit_code")
    return int(exit_code) if isinstance(exit_code, int) else 1


class Sam3DBodyPersistentWorker:
    """Bootstraps one estimator once, then serves batch jobs over a Unix socket."""

    def __init__(
        self,
        *,
        fast_sam_repo: Path,
        checkpoint_dir: Path,
        detector_name: str | None,
        detector_model: str,
        fov_name: str,
        bootstrap_requests_path: Path,
        socket_path: str,
        ready_path: str | None = None,
        idle_timeout_s: float = 1200.0,
        max_consecutive_job_crashes: int = 2,
        accept_poll_s: float = 5.0,
    ) -> None:
        self.fast_sam_repo = Path(fast_sam_repo)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.detector_name = detector_name
        self.detector_model = detector_model
        self.fov_name = fov_name
        self.bootstrap_requests_path = Path(bootstrap_requests_path)
        self.socket_path = str(socket_path)
        self.ready_path = ready_path
        self.idle_timeout_s = float(idle_timeout_s)
        self.max_consecutive_job_crashes = int(max_consecutive_job_crashes)
        self.accept_poll_s = float(accept_poll_s)

        self.code_identity = _hash_code_identity()
        self.estimator: Any = None
        self.runtime_environment: dict[str, Any] | None = None
        self.mhr_correctives: dict[str, Any] | None = None
        self.faces: Any = None
        self.compile_warmup: dict[str, Any] | None = None
        self.fingerprint: str | None = None
        self.bootstrap_timing_summary: dict[str, Any] | None = None
        self.consecutive_job_crashes = 0
        self.healthy = True
        self._should_exit = False
        self.last_activity_monotonic = time.monotonic()

    def bootstrap(self) -> dict[str, Any]:
        payload = json.loads(self.bootstrap_requests_path.read_text(encoding="utf-8"))
        batch_payload = _parse_batch_payload(payload)
        optimization = dict(batch_payload["optimization"])
        batch_payload["optimization"] = optimization
        body_input_size = normalize_body_input_size(optimization.get("sam3d_body_input_size_px"))
        optimization["sam3d_body_input_size_px"] = body_input_size

        # Ordering constraint (see run_sam3dbody_batch.py main()): upstream
        # FastSAM modules read these env vars during import/model
        # construction, so this must run before _load_setup_sam_3d_body()/
        # _setup_estimator(). Once bootstrap finishes, the resolved
        # environment is baked into this process for its whole lifetime --
        # exactly why later jobs must fingerprint-match this bootstrap config.
        runtime_environment = _configure_runtime_environment(optimization)
        if body_input_size is not None:
            os.environ["IMG_SIZE"] = str(body_input_size)

        path_errors: list[str] = []
        for request in batch_payload["requests"]:
            path_errors.extend(_runtime_path_errors(Path(request["image"]), self.fast_sam_repo, self.checkpoint_dir))
        if path_errors:
            raise RuntimeError("; ".join(path_errors))

        resolved_detector_name = _detector_name(
            self.detector_name, [bbox for request in batch_payload["requests"] for bbox in request["bboxes"]]
        )

        timer = _TimingRecorder()
        with timer.span("model_setup_load"):
            setup_sam_3d_body = _load_setup_sam_3d_body(self.fast_sam_repo)
            estimator = _setup_estimator(
                setup_sam_3d_body,
                checkpoint_dir=self.checkpoint_dir.resolve(),
                detector_name=resolved_detector_name,
                detector_model=self.detector_model,
                fov_name=self.fov_name,
            )
        with timer.span("compile_warmup"):
            compile_warmup = _warmup_static_clip_intrinsics(
                estimator,
                clip_intrinsics=batch_payload["clip_intrinsics"],
                optimization=optimization,
                timing=timer,
            )

        self.estimator = estimator
        self.runtime_environment = runtime_environment
        self.mhr_correctives = _detect_mhr_correctives_active(estimator)
        self.faces = _json_safe(getattr(estimator, "faces", None))
        self.compile_warmup = compile_warmup
        self.fingerprint = compute_config_fingerprint(
            checkpoint_dir=self.checkpoint_dir,
            detector_name=self.detector_name or "",
            detector_model=self.detector_model or "",
            fov_name=self.fov_name or "",
            optimization=optimization,
            clip_intrinsics=batch_payload["clip_intrinsics"],
            code_identity=self.code_identity,
        )
        self.bootstrap_timing_summary = _sam3d_batch_timing_summary(
            timer.events, person_frame_count=len(batch_payload["requests"])
        )
        self.last_activity_monotonic = time.monotonic()

        ready_payload = {
            "schema_version": 1,
            "artifact_type": WORKER_READY_ARTIFACT_TYPE,
            "pid": os.getpid(),
            "fingerprint": self.fingerprint,
            "bootstrap_timing_summary": self.bootstrap_timing_summary,
            "socket_path": self.socket_path,
        }
        if self.ready_path is not None:
            _write_json_payload(Path(self.ready_path), ready_payload)
        return ready_payload

    def _job_fingerprint(self, batch_payload: Mapping[str, Any]) -> str:
        return compute_config_fingerprint(
            checkpoint_dir=self.checkpoint_dir,
            detector_name=self.detector_name or "",
            detector_model=self.detector_model or "",
            fov_name=self.fov_name or "",
            optimization=batch_payload["optimization"],
            clip_intrinsics=batch_payload["clip_intrinsics"],
            code_identity=self.code_identity,
        )

    def _cuda_canary_healthy(self) -> bool:
        try:
            import torch
        except ImportError:
            return True
        try:
            if not torch.cuda.is_available():
                return True
            torch.cuda.synchronize()
            probe = torch.zeros(4, device="cuda")
            float((probe + 1).sum().item())
            return True
        except Exception:  # noqa: BLE001 - any CUDA-context failure means "not healthy"
            return False

    def handle_job(self, job: Mapping[str, Any]) -> dict[str, Any]:
        try:
            requests_path = Path(str(job["requests_path"]))
            out_path = Path(str(job["out_path"]))
            chunk_dir_raw = job.get("chunk_dir")
            chunk_dir = Path(str(chunk_dir_raw)) if chunk_dir_raw else _default_chunk_dir(out_path)
            chunk_format = str(job.get("chunk_format") or "pickle")
            no_monolithic_output = bool(job.get("no_monolithic_output", False))
            bucket_size = job.get("bucket_size")

            payload = json.loads(requests_path.read_text(encoding="utf-8"))
            batch_payload = _parse_batch_payload(payload)
            optimization = _apply_bucket_size_override(batch_payload["optimization"], bucket_size=bucket_size)
            batch_payload["optimization"] = optimization
            body_input_size = normalize_body_input_size(
                job.get("body_input_size") or optimization.get("sam3d_body_input_size_px")
            )
            optimization["sam3d_body_input_size_px"] = body_input_size
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return {"status": "bad_request", "exit_code": 2, "detail": f"{type(exc).__name__}: {exc}"}

        job_fingerprint = self._job_fingerprint(batch_payload)
        if job_fingerprint != self.fingerprint:
            return {
                "status": "fingerprint_mismatch",
                "exit_code": 3,
                "server_fingerprint": self.fingerprint,
                "job_fingerprint": job_fingerprint,
            }

        if not self.healthy or not self._cuda_canary_healthy():
            self.healthy = False
            self._should_exit = True
            return {
                "status": "worker_unhealthy",
                "exit_code": 4,
                "detail": "CUDA context canary failed or worker was already marked unhealthy",
            }

        requests = batch_payload["requests"]
        job_timer = _TimingRecorder()
        try:
            _run_batch_inference_and_write(
                estimator=self.estimator,
                requests=requests,
                batch_payload=batch_payload,
                optimization=optimization,
                faces=self.faces,
                runtime_environment=self.runtime_environment or {},
                mhr_correctives=self.mhr_correctives or {},
                compile_warmup=self.compile_warmup or {},
                out_path=out_path,
                chunk_dir=chunk_dir,
                chunk_format=chunk_format,
                write_monolithic=not no_monolithic_output,
                timer=job_timer,
            )
        except Exception as exc:  # noqa: BLE001 - mirrors run_sam3dbody_batch.py main()'s own catch-all
            _write_failure_output(
                out_path,
                exc,
                request_count=len(requests),
                optimization=optimization,
                compile_warmup=self.compile_warmup,
            )
            self.consecutive_job_crashes += 1
            result: dict[str, Any] = {
                "status": "error",
                "exit_code": 1,
                "error": f"{type(exc).__name__}: {exc}",
                "consecutive_job_crashes": self.consecutive_job_crashes,
            }
            if self.consecutive_job_crashes >= self.max_consecutive_job_crashes:
                result["worker_exiting"] = True
                self._should_exit = True
            return result

        timing_summary = _sam3d_batch_timing_summary(job_timer.events, person_frame_count=len(requests))
        _write_json_payload(_timing_sidecar_path(out_path), timing_summary)
        self.consecutive_job_crashes = 0
        return {"status": "ok", "exit_code": 0, "out_path": str(out_path), "timing_summary": timing_summary}

    def serve_forever(self) -> dict[str, Any]:
        socket_path = self.socket_path
        try:
            os.unlink(socket_path)
        except FileNotFoundError:
            pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)
        server.settimeout(self.accept_poll_s)
        exit_reason = "idle_timeout"
        try:
            while True:
                if self._should_exit:
                    exit_reason = "job_failure_limit_or_unhealthy"
                    break
                idle_for = time.monotonic() - self.last_activity_monotonic
                if idle_for > self.idle_timeout_s:
                    exit_reason = "idle_timeout"
                    break
                try:
                    conn, _addr = server.accept()
                except socket.timeout:
                    continue
                with conn:
                    try:
                        job = _recv_message(conn)
                        response = self.handle_job(job)
                    except Exception as exc:  # noqa: BLE001 - a protocol-level failure still counts as a crash
                        self.consecutive_job_crashes += 1
                        response = {
                            "status": "error",
                            "exit_code": 1,
                            "error": f"protocol_error: {type(exc).__name__}: {exc}",
                            "consecutive_job_crashes": self.consecutive_job_crashes,
                        }
                        if self.consecutive_job_crashes >= self.max_consecutive_job_crashes:
                            response["worker_exiting"] = True
                            self._should_exit = True
                    try:
                        _send_message(conn, response)
                    except OSError:
                        pass
                self.last_activity_monotonic = time.monotonic()
        finally:
            server.close()
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass
        exit_summary = {
            "schema_version": 1,
            "artifact_type": WORKER_EXIT_ARTIFACT_TYPE,
            "reason": exit_reason,
            "consecutive_job_crashes": self.consecutive_job_crashes,
            "healthy": self.healthy,
        }
        if self.ready_path is not None:
            _write_json_payload(Path(str(self.ready_path) + ".exit.json"), exit_summary)
        return exit_summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental persistent SAM-3D-Body worker (body_overhead_20260712, Lever A). "
            "Default-off, opt-in only via SAM3DBODY_PERSISTENT_WORKER_SOCKET / "
            "RemoteConfig.sam3dbody_persistent_worker_socket. VERIFIED=0."
        )
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    serve = sub.add_parser("serve", help="Load the model once and serve batch jobs over a local Unix socket.")
    serve.add_argument("--fast-sam-repo", type=Path, required=True)
    serve.add_argument("--checkpoint-dir", type=Path, required=True)
    serve.add_argument("--detector-model", default="")
    serve.add_argument("--detector-name", default=None)
    serve.add_argument("--fov-name", default="")
    serve.add_argument(
        "--bootstrap-requests",
        type=Path,
        required=True,
        help="A requests.json (same schema as run_sam3dbody_batch.py --requests) used only to "
        "resolve optimization/clip_intrinsics/detector defaults and to perform the real "
        "model-load + compile warmup once at startup.",
    )
    serve.add_argument("--socket-path", required=True)
    serve.add_argument("--ready-path", default=None, help="Optional path to write the bootstrap-ready JSON to.")
    serve.add_argument("--idle-timeout-s", type=float, default=1200.0)
    serve.add_argument("--max-consecutive-job-crashes", type=int, default=2)
    serve.add_argument("--accept-poll-s", type=float, default=5.0)

    client = sub.add_parser("client", help="Submit one batch job to an already-running persistent worker.")
    client.add_argument("--socket-path", required=True)
    client.add_argument("--requests", type=Path, required=True)
    client.add_argument("--out", type=Path, required=True)
    client.add_argument("--chunk-dir", type=Path, default=None)
    client.add_argument("--chunk-format", default="pickle")
    client.add_argument("--no-monolithic-output", action="store_true")
    client.add_argument("--bucket-size", type=int, default=None)
    client.add_argument("--body-input-size", type=int, default=None)
    client.add_argument("--connect-timeout-s", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode == "client":
        return submit_job_via_client(
            socket_path=args.socket_path,
            requests_path=args.requests,
            out_path=args.out,
            chunk_dir=args.chunk_dir,
            chunk_format=args.chunk_format,
            no_monolithic_output=args.no_monolithic_output,
            bucket_size=args.bucket_size,
            body_input_size=args.body_input_size,
            connect_timeout_s=args.connect_timeout_s,
        )

    worker = Sam3DBodyPersistentWorker(
        fast_sam_repo=args.fast_sam_repo,
        checkpoint_dir=args.checkpoint_dir,
        detector_name=args.detector_name,
        detector_model=args.detector_model,
        fov_name=args.fov_name,
        bootstrap_requests_path=args.bootstrap_requests,
        socket_path=args.socket_path,
        ready_path=args.ready_path,
        idle_timeout_s=args.idle_timeout_s,
        max_consecutive_job_crashes=args.max_consecutive_job_crashes,
        accept_poll_s=args.accept_poll_s,
    )
    ready = worker.bootstrap()
    print(json.dumps(ready, sort_keys=True), flush=True)
    exit_summary = worker.serve_forever()
    print(json.dumps(exit_summary, sort_keys=True), flush=True)
    return 0 if exit_summary.get("reason") == "idle_timeout" else 1


if __name__ == "__main__":
    raise SystemExit(main())
